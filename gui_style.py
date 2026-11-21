# gui_style.py
import imgui

def setup_style():
    style = imgui.get_style()
    
    # --- Premium UI Dimensions & Rounding (Proto.io Inspired) ---
    style.window_rounding = 8.0       # Softer, modern windows
    style.child_rounding = 6.0
    style.frame_rounding = 6.0        # Rounder buttons and inputs
    style.grab_rounding = 6.0
    style.popup_rounding = 8.0
    style.scrollbar_rounding = 12.0
    
    # Generous spacing for a clean, uncluttered layout
    style.frame_padding = (10, 6)     
    style.item_spacing = (10, 8)      
    style.window_padding = (12, 12)
    style.window_border_size = 1.0    # Clean, subtle borders
    
    # --- Colors (Premium Dark Theme with Vibrant Accents) ---
    def c(r, g, b, a=1.0): return (r/255.0, g/255.0, b/255.0, a)

    colors = style.colors
    
    # Text (High contrast off-white for readability)
    colors[imgui.COLOR_TEXT]                  = c(245, 245, 250)
    colors[imgui.COLOR_TEXT_DISABLED]         = c(140, 140, 145)
    
    # Backgrounds (Deep, layered slate tones)
    colors[imgui.COLOR_WINDOW_BACKGROUND]     = c(26, 26, 30, 0.98) 
    colors[imgui.COLOR_CHILD_BACKGROUND]      = c(32, 32, 36, 0.0)
    colors[imgui.COLOR_POPUP_BACKGROUND]      = c(30, 30, 34, 0.98)
    
    # Borders (Subtle separation)
    colors[imgui.COLOR_BORDER]                = c(65, 65, 70, 0.6)
    colors[imgui.COLOR_BORDER_SHADOW]         = c(0, 0, 0, 0.15)
    
    # Frames (Inputs, Checkboxes, Base Buttons)
    colors[imgui.COLOR_FRAME_BACKGROUND]      = c(42, 42, 48)
    colors[imgui.COLOR_FRAME_BACKGROUND_HOVERED] = c(58, 58, 65)
    colors[imgui.COLOR_FRAME_BACKGROUND_ACTIVE]  = c(75, 75, 82)
    
    # Titles
    colors[imgui.COLOR_TITLE_BACKGROUND]      = c(22, 22, 25)
    colors[imgui.COLOR_TITLE_BACKGROUND_ACTIVE] = c(22, 22, 25)
    colors[imgui.COLOR_TITLE_BACKGROUND_COLLAPSED] = c(22, 22, 25, 0.7)
    
    # Menu Bar
    colors[imgui.COLOR_MENUBAR_BACKGROUND]    = c(20, 20, 23)
    
    # Scrollbar
    colors[imgui.COLOR_SCROLLBAR_BACKGROUND]  = c(24, 24, 28)
    colors[imgui.COLOR_SCROLLBAR_GRAB]        = c(70, 70, 75)
    colors[imgui.COLOR_SCROLLBAR_GRAB_HOVERED] = c(90, 90, 95)
    colors[imgui.COLOR_SCROLLBAR_GRAB_ACTIVE] = c(110, 110, 115)
    
    # --- Signature Accents (Vibrant Modern Blue) ---
    ACCENT = c(10, 132, 255)         
    ACCENT_HOVER = c(45, 150, 255)
    ACCENT_ACTIVE = c(0, 110, 220)
    
    colors[imgui.COLOR_CHECK_MARK]            = ACCENT
    
    colors[imgui.COLOR_SLIDER_GRAB]           = ACCENT
    colors[imgui.COLOR_SLIDER_GRAB_ACTIVE]    = ACCENT_ACTIVE
    
    # Base Interactive Buttons
    colors[imgui.COLOR_BUTTON]                = c(48, 48, 54)
    colors[imgui.COLOR_BUTTON_HOVERED]        = c(68, 68, 75)
    colors[imgui.COLOR_BUTTON_ACTIVE]         = ACCENT_ACTIVE
    
    # Headers (CollapsingHeader, TreeNodes)
    colors[imgui.COLOR_HEADER]                = c(48, 48, 54)
    colors[imgui.COLOR_HEADER_HOVERED]        = c(68, 68, 75)
    colors[imgui.COLOR_HEADER_ACTIVE]         = ACCENT
    
    # Miscellaneous Grabs & Separators
    colors[imgui.COLOR_SEPARATOR]             = c(65, 65, 70, 0.6)
    colors[imgui.COLOR_RESIZE_GRIP]           = c(65, 65, 70, 0.5)
    colors[imgui.COLOR_RESIZE_GRIP_HOVERED]   = ACCENT
    colors[imgui.COLOR_RESIZE_GRIP_ACTIVE]    = ACCENT_ACTIVE
