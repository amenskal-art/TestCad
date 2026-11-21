# tools/camera_identity.py
# Camera identity, enumeration and "lock to one physical camera" support.
#
# OpenCV's VideoCapture(index) only knows positional indices (0,1,2...), which
# shift around depending on plug order and what else is attached. To recognize
# ONE specific physical camera and refuse every other one, we have to read the
# device's hardware identifiers from the OS and map them back to an OpenCV
# index. That is inherently platform specific:
#
#   Windows : DirectShow friendly names (via pygrabber if present) + WMI/registry
#             for USB VID/PID/serial. DirectShow enumeration order matches the
#             cv2.CAP_DSHOW index order, which lets us line names up with indices.
#   Linux   : /sys/class/video4linux/video*/ for the name + the matching
#             /dev/videoN index, and udev (pyudev) or sysfs for VID/PID/serial.
#
# Everything here degrades gracefully: if the rich identifiers are unavailable
# we still produce a stable-ish fingerprint from whatever we can read (name +
# index), so the feature keeps working in a reduced form instead of crashing.

import os
import json
import sys
import re
import subprocess


# ----------------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------------

class CameraInfo:
    """One detected camera device."""
    __slots__ = ("index", "name", "vid", "pid", "serial", "is_builtin", "path")

    def __init__(self, index, name="", vid=None, pid=None, serial=None,
                 is_builtin=False, path=None):
        self.index = index
        self.name = name or f"Camera {index}"
        self.vid = vid                 # USB Vendor ID  (e.g. "046d")
        self.pid = pid                 # USB Product ID (e.g. "0825")
        self.serial = serial           # USB serial string if exposed
        self.is_builtin = is_builtin   # best-effort guess (laptop integrated cam)
        self.path = path               # OS path (e.g. /dev/video0) where relevant

    def fingerprint(self):
        """A stable identity string for THIS physical camera.

        Prefers the hardware identifiers (VID/PID/serial). Falls back to the
        device name when those aren't available. Index is deliberately NOT part
        of the fingerprint — it is the thing we expect to move around.
        """
        if self.vid and self.pid and self.serial:
            return f"usb:{self.vid}:{self.pid}:{self.serial}".lower()
        if self.vid and self.pid:
            return f"usb:{self.vid}:{self.pid}".lower()
        # Last resort: name-based. Not unique across identical models, but far
        # better than a bare index.
        clean = re.sub(r"\s+", " ", (self.name or "").strip()).lower()
        return f"name:{clean}" if clean else f"index:{self.index}"

    def matches_fingerprint(self, fp):
        if not fp:
            return False
        return self.fingerprint() == fp

    def to_dict(self):
        return {
            "index": self.index, "name": self.name, "vid": self.vid,
            "pid": self.pid, "serial": self.serial,
            "is_builtin": self.is_builtin, "path": self.path,
        }

    def __repr__(self):
        tag = "builtin" if self.is_builtin else "external"
        return (f"<CameraInfo idx={self.index} '{self.name}' "
                f"{self.vid}:{self.pid} {tag}>")


# Heuristics for guessing a built-in (integrated/laptop) webcam by name.
_BUILTIN_NAME_HINTS = (
    "integrated", "built-in", "builtin", "internal", "facetime", "hd webcam",
    "ir camera", "infrared", "front camera", "ec_camera", "easycamera",
)


def _looks_builtin(name):
    n = (name or "").lower()
    return any(h in n for h in _BUILTIN_NAME_HINTS)


# ----------------------------------------------------------------------------
# Windows enumeration
# ----------------------------------------------------------------------------

def _enumerate_windows():
    cams = []

    # 1) DirectShow friendly names, in the SAME order OpenCV's CAP_DSHOW uses.
    names = []
    try:
        from pygrabber.dshow_graph import FilterGraph
        names = FilterGraph().get_input_devices()  # list[str], index == position
    except Exception:
        names = []

    # 2) USB VID/PID/serial for imaging devices, via WMI/PowerShell. We match
    #    these onto the DirectShow names by best-effort name containment.
    usb_devices = _windows_usb_imaging_devices()

    if names:
        for idx, nm in enumerate(names):
            vid = pid = serial = None
            # try to find a usb device whose friendly name relates to nm
            for ud in usb_devices:
                if _name_overlap(nm, ud.get("name", "")):
                    vid = ud.get("vid"); pid = ud.get("pid"); serial = ud.get("serial")
                    break
            cams.append(CameraInfo(index=idx, name=nm, vid=vid, pid=pid,
                                   serial=serial, is_builtin=_looks_builtin(nm)))
        return cams

    # No pygrabber: build the list purely from the PowerShell PnP data. We
    # deliberately do NOT brute-force open camera indices here — opening
    # devices with cv2.VideoCapture just to enumerate them is what produced
    # the repeated "VIDEOIO(DSHOW): ... can't be used to capture by index"
    # warnings AND fought the live stream for the device. DirectShow's
    # enumeration order matches the index order used by CAP_DSHOW, so the
    # PnP order is a reasonable index assignment.
    for idx, ud in enumerate(usb_devices):
        cams.append(CameraInfo(index=idx, name=ud.get("name", f"Camera {idx}"),
                               vid=ud.get("vid"), pid=ud.get("pid"),
                               serial=ud.get("serial"),
                               is_builtin=_looks_builtin(ud.get("name", ""))))
    return cams


def _windows_usb_imaging_devices():
    """Return [{name, vid, pid, serial}] for USB imaging devices via PowerShell.

    Uses PnP device data. PNPDeviceID looks like:
        USB\\VID_046D&PID_0825\\<serial-or-instance>
    """
    devices = []
    ps = (
        "Get-PnpDevice -Class Camera,Image -PresentOnly "
        "| Select-Object FriendlyName,InstanceId "
        "| ConvertTo-Json -Compress"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True, text=True, timeout=8,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout.strip()
        if not out:
            return devices
        data = json.loads(out)
        if isinstance(data, dict):
            data = [data]
        for d in data:
            inst = d.get("InstanceId", "") or ""
            m = re.search(r"VID_([0-9A-Fa-f]{4})&PID_([0-9A-Fa-f]{4})\\?(.*)?", inst)
            vid = pid = serial = None
            if m:
                vid = m.group(1).lower()
                pid = m.group(2).lower()
                tail = (m.group(3) or "").strip("\\")
                # A real serial has no '&' (instance paths contain '&xx&xx').
                if tail and "&" not in tail:
                    serial = tail
            devices.append({"name": d.get("FriendlyName", ""),
                            "vid": vid, "pid": pid, "serial": serial})
    except Exception:
        pass
    return devices


def _name_overlap(a, b):
    a = (a or "").lower(); b = (b or "").lower()
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    # token overlap
    at = set(re.findall(r"[a-z0-9]+", a))
    bt = set(re.findall(r"[a-z0-9]+", b))
    return len(at & bt) >= 2


# ----------------------------------------------------------------------------
# Linux enumeration
# ----------------------------------------------------------------------------

def _enumerate_linux():
    cams = []
    base = "/sys/class/video4linux"
    if not os.path.isdir(base):
        # No v4l sysfs (headless container etc.) -> probe indices only.
        for idx in _probe_indices(max_probe=8):
            cams.append(CameraInfo(index=idx, name=f"Camera {idx}",
                                   path=f"/dev/video{idx}"))
        return cams

    seen_fp = set()
    for node in sorted(os.listdir(base)):
        if not node.startswith("video"):
            continue
        try:
            idx = int(node.replace("video", ""))
        except ValueError:
            continue
        ndir = os.path.join(base, node)
        name = _read_file(os.path.join(ndir, "name")) or f"Camera {idx}"

        vid = pid = serial = None
        # Walk up the device symlink to the USB device dir holding idVendor etc.
        devlink = os.path.join(ndir, "device")
        usbdir = _find_usb_parent(os.path.realpath(devlink)) if os.path.islink(devlink) else None
        if usbdir:
            vid = _read_file(os.path.join(usbdir, "idVendor"))
            pid = _read_file(os.path.join(usbdir, "idProduct"))
            serial = _read_file(os.path.join(usbdir, "serial"))

        info = CameraInfo(index=idx, name=name, vid=vid, pid=pid, serial=serial,
                          is_builtin=_looks_builtin(name), path=f"/dev/video{idx}")

        # v4l exposes several /dev/videoN per physical camera (capture +
        # metadata). Collapse duplicates by fingerprint, keep the lowest index.
        fp = info.fingerprint()
        if fp in seen_fp:
            continue
        seen_fp.add(fp)
        cams.append(info)
    return cams


def _find_usb_parent(path):
    """Climb a /sys realpath until we hit a dir that has idVendor/idProduct."""
    cur = path
    for _ in range(12):
        if not cur or cur == "/":
            break
        if os.path.exists(os.path.join(cur, "idVendor")) and \
           os.path.exists(os.path.join(cur, "idProduct")):
            return cur
        cur = os.path.dirname(cur)
    return None


def _read_file(p):
    try:
        with open(p, "r") as f:
            return f.read().strip()
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Shared
# ----------------------------------------------------------------------------

def _probe_indices(max_probe=8):
    """Return indices that actually open. Used only as a last resort because
    opening cameras just to enumerate them is slow and can fight the live
    stream for the device."""
    found = []
    try:
        import cv2
    except Exception:
        return found
    for i in range(max_probe):
        cap = None
        try:
            if os.name == "nt":
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(i)
            if cap is not None and cap.isOpened():
                found.append(i)
        except Exception:
            pass
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
    return found


def pygrabber_available():
    """True if pygrabber is importable (Windows). When present we get the
    authoritative DirectShow device order, so OpenCV indices map correctly to
    physical cameras. Without it, index assignment on Windows is a best-effort
    guess from PnP enumeration order."""
    if os.name != "nt":
        return True  # not needed off Windows
    try:
        import pygrabber.dshow_graph  # noqa: F401
        return True
    except Exception:
        return False


def enumerate_cameras():
    """Return a list[CameraInfo] of currently attached cameras (best effort)."""
    try:
        if os.name == "nt":
            return _enumerate_windows()
        if sys.platform == "darwin":
            # macOS: no rich USB mapping without extra deps; probe + name guess.
            cams = []
            for idx in _probe_indices(max_probe=6):
                cams.append(CameraInfo(index=idx, name=f"Camera {idx}"))
            return cams
        return _enumerate_linux()
    except Exception:
        # Never let enumeration crash the app.
        return []


def list_external_cameras():
    """Cameras that look like EXTERNAL/wired devices (not the integrated cam).

    A device is treated as external when it isn't flagged built-in by name AND
    it exposes USB VID/PID (integrated cams frequently don't, or are flagged).
    Falls back to 'anything not name-flagged builtin'.
    """
    cams = enumerate_cameras()
    ext = [c for c in cams if (not c.is_builtin) and c.vid and c.pid]
    if ext:
        return ext
    return [c for c in cams if not c.is_builtin]


def find_by_fingerprint(fp):
    """Return the CameraInfo currently matching fingerprint fp, or None."""
    if not fp:
        return None
    for c in enumerate_cameras():
        if c.matches_fingerprint(fp):
            return c
    return None


# ----------------------------------------------------------------------------
# Persistence of the locked / authorized camera
# ----------------------------------------------------------------------------

class LockedCameraStore:
    """Persists the single authorized camera's fingerprint + a friendly label.

    'First plug wins': the first external camera we detect is recorded here and
    becomes THE camera. After that the app only ever connects to a device whose
    fingerprint matches; everything else is rejected.
    """

    def __init__(self, path=None):
        self.path = path or os.path.expanduser("~/.iqscanner_locked_camera.json")
        self.fingerprint = None
        self.label = None
        self.vid = None
        self.pid = None
        self.serial = None
        self.load()

    def load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, "r") as f:
                    d = json.load(f)
                self.fingerprint = d.get("fingerprint")
                self.label = d.get("label")
                self.vid = d.get("vid")
                self.pid = d.get("pid")
                self.serial = d.get("serial")
        except Exception:
            pass

    def save(self):
        try:
            with open(self.path, "w") as f:
                json.dump({
                    "fingerprint": self.fingerprint, "label": self.label,
                    "vid": self.vid, "pid": self.pid, "serial": self.serial,
                }, f)
        except Exception:
            pass

    def is_set(self):
        return bool(self.fingerprint)

    def lock_to(self, cam):
        """Record this CameraInfo as the one authorized camera."""
        self.fingerprint = cam.fingerprint()
        self.label = cam.name
        self.vid = cam.vid
        self.pid = cam.pid
        self.serial = cam.serial
        self.save()

    def clear(self):
        self.fingerprint = None
        self.label = None
        self.vid = self.pid = self.serial = None
        try:
            if os.path.exists(self.path):
                os.remove(self.path)
        except Exception:
            pass

    def matching_camera(self):
        """Currently-attached CameraInfo matching the lock, or None."""
        if not self.is_set():
            return None
        return find_by_fingerprint(self.fingerprint)
