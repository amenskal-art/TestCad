# tool_system.py - NEW FILE
import abc
from typing import Dict, Type, Optional
import imgui
import glfw

class BaseTool(abc.ABC):
    """Abstract base class for all tools."""
    
    def __init__(self, app):
        self.app = app
        self.name = "Unnamed Tool"
        self.icon_key = None
        self.hotkey = None  # String like '1', '2', etc.
        self.is_active = False
        self.tool_id = -1
        
    def on_activate(self):
        """Called when tool is selected."""
        pass
        
    def on_deactivate(self):
        """Called when tool is deselected."""
        pass
        
    def draw_options(self):
        """Draw tool-specific options panel."""
        imgui.text(f"No options for {self.name}")
        
    def on_mouse_button(self, button, action, mods, x, y):
        """Handle mouse button events."""
        return False
        
    def on_mouse_move(self, x, y):
        """Handle mouse movement."""
        return False
        
    def on_key(self, key, scancode, action, mods):
        """Handle keyboard events."""
        return False
        
    def update(self):
        """Called every frame while tool is active."""
        pass
        
    def render_3d(self):
        """Render 3D elements (gizmos, cursors, etc.)."""
        pass
        
    def render_2d_overlay(self):
        """Render 2D overlay elements."""
        pass

class ToolManager:
    """Manages registration and lifecycle of all tools."""
    
    def __init__(self, app):
        self.app = app
        self.tools: Dict[int, BaseTool] = {}
        self.active_tool_id = 0
        self.tool_classes = {}
        
    def register_tool(self, tool_id: int, tool_class: Type[BaseTool]):
        """Register a new tool class."""
        self.tool_classes[tool_id] = tool_class
        
    def initialize_tools(self):
        """Create instances of all registered tools."""
        for tool_id, tool_class in self.tool_classes.items():
            tool_instance = tool_class(self.app)
            tool_instance.tool_id = tool_id
            self.tools[tool_id] = tool_instance
            
    def set_active_tool(self, tool_id: int):
        """Switch to a different tool."""
        if tool_id not in self.tools:
            print(f"Tool {tool_id} not found")
            return
            
        # Deactivate current tool
        if self.active_tool_id in self.tools:
            self.tools[self.active_tool_id].on_deactivate()
            self.tools[self.active_tool_id].is_active = False
            
        # Activate new tool
        self.active_tool_id = tool_id
        self.tools[tool_id].on_activate()
        self.tools[tool_id].is_active = True
            
    def get_active_tool(self) -> Optional[BaseTool]:
        """Get the currently active tool."""
        return self.tools.get(self.active_tool_id)
        
    def handle_event(self, event_name: str, *args, **kwargs):
        """Route events to active tool."""
        tool = self.get_active_tool()
        if tool and hasattr(tool, f"on_{event_name}"):
            return getattr(tool, f"on_{event_name}")(*args, **kwargs)
        return False
