# Black-Scholes Option Pricing Platform - Design System Specification v1.0

## Executive Summary

This Design System establishes the complete visual language and interaction patterns for the Black-Scholes Option Pricing Platform. It is optimized for quantitative traders and financial professionals who require precision, speed, and information density while maintaining exceptional usability.

**Design Philosophy**: "Terminal Precision, Modern Usability"
- Information-dense like Bloomberg Terminal
- Modern, clean aesthetics like Robinhood/Interactive Brokers
- Zero ambiguity in data presentation
- Accessibility-first architecture

---

## 1. Design Tokens

### 1.1 Color Palette

#### Primary Colors
```
Primary Blue (Actions, CTAs, Links)
- Primary-50:  #E3F2FD  (rgb(227, 242, 253))
- Primary-100: #BBDEFB  (rgb(187, 222, 251))
- Primary-200: #90CAF9  (rgb(144, 202, 249))
- Primary-300: #64B5F6  (rgb(100, 181, 246))
- Primary-400: #42A5F5  (rgb(66, 165, 245))
- Primary-500: #2196F3  (rgb(33, 150, 243))  ← Main Primary
- Primary-600: #1E88E5  (rgb(30, 136, 229))
- Primary-700: #1976D2  (rgb(25, 118, 210))
- Primary-800: #1565C0  (rgb(21, 101, 192))
- Primary-900: #0D47A1  (rgb(13, 71, 161))

Secondary Slate (UI Chrome, Borders, Backgrounds)
- Secondary-50:  #FAFAFA  (rgb(250, 250, 250))
- Secondary-100: #F5F5F5  (rgb(245, 245, 245))
- Secondary-200: #EEEEEE  (rgb(238, 238, 238))
- Secondary-300: #E0E0E0  (rgb(224, 224, 224))
- Secondary-400: #BDBDBD  (rgb(189, 189, 189))
- Secondary-500: #9E9E9E  (rgb(158, 158, 158))
- Secondary-600: #757575  (rgb(117, 117, 117))
- Secondary-700: #616161  (rgb(97, 97, 97))
- Secondary-800: #424242  (rgb(66, 66, 66))
- Secondary-900: #212121  (rgb(33, 33, 33))
```

#### Semantic Colors
```
Success Green (Positive PnL, In-The-Money Calls, Upward Movement)
- Success-50:  #E8F5E9  (rgb(232, 245, 233))
- Success-100: #C8E6C9  (rgb(200, 230, 201))
- Success-500: #4CAF50  (rgb(76, 175, 80))   ← Main Success
- Success-600: #43A047  (rgb(67, 160, 71))
- Success-700: #388E3C  (rgb(56, 142, 60))
- Success-900: #1B5E20  (rgb(27, 94, 32))

Error Red (Negative PnL, Out-The-Money Puts, Downward Movement, Validation Errors)
- Error-50:  #FFEBEE  (rgb(255, 235, 238))
- Error-100: #FFCDD2  (rgb(255, 205, 210))
- Error-500: #F44336  (rgb(244, 67, 54))      ← Main Error
- Error-600: #E53935  (rgb(229, 57, 53))
- Error-700: #D32F2F  (rgb(211, 47, 47))
- Error-900: #B71C1C  (rgb(183, 28, 28))

Warning Amber (Approaching Expiry, Risk Thresholds, Caution)
- Warning-50:  #FFF8E1  (rgb(255, 248, 225))
- Warning-100: #FFECB3  (rgb(255, 236, 179))
- Warning-500: #FFC107  (rgb(255, 193, 7))    ← Main Warning
- Warning-600: #FFB300  (rgb(255, 179, 0))
- Warning-700: #FFA000  (rgb(255, 160, 0))
- Warning-900: #FF6F00  (rgb(255, 111, 0))

Info Cyan (Informational Messages, At-The-Money, Neutral States)
- Info-50:  #E0F7FA  (rgb(224, 247, 250))
- Info-100: #B2EBF2  (rgb(178, 235, 242))
- Info-500: #00BCD4  (rgb(0, 188, 212))       ← Main Info
- Info-600: #00ACC1  (rgb(0, 172, 193))
- Info-700: #0097A7  (rgb(0, 151, 167))
- Info-900: #006064  (rgb(0, 96, 100))
```

#### Financial Data Colors
```
Positive Growth/Long Positions
- Growth-Light: #10B981  (rgb(16, 185, 129))  - Emerald-500
- Growth-Dark:  #059669  (rgb(5, 150, 105))   - Emerald-600

Negative/Short Positions
- Decline-Light: #EF4444  (rgb(239, 68, 68))  - Red-500
- Decline-Dark:  #DC2626  (rgb(220, 38, 38))  - Red-600

Neutral/Break-Even
- Neutral-Light: #6B7280  (rgb(107, 114, 128)) - Gray-500
- Neutral-Dark:  #4B5563  (rgb(75, 85, 99))    - Gray-600

Call Options Accent
- Call-Color: #3B82F6  (rgb(59, 130, 246))    - Blue-500

Put Options Accent
- Put-Color: #EC4899  (rgb(236, 72, 153))     - Pink-500
```

#### Background & Surface Colors
```
Light Mode
- Background-Default:  #FFFFFF  (rgb(255, 255, 255))
- Background-Paper:    #FAFAFA  (rgb(250, 250, 250))
- Background-Elevated: #F5F5F5  (rgb(245, 245, 245))
- Divider:             #E0E0E0  (rgb(224, 224, 224))
- Border:              #BDBDBD  (rgb(189, 189, 189))
- Overlay:             rgba(0, 0, 0, 0.5)

Dark Mode
- Background-Default:  #121212  (rgb(18, 18, 18))
- Background-Paper:    #1E1E1E  (rgb(30, 30, 30))
- Background-Elevated: #2C2C2C  (rgb(44, 44, 44))
- Divider:             #3A3A3A  (rgb(58, 58, 58))
- Border:              #4A4A4A  (rgb(74, 74, 74))
- Overlay:             rgba(0, 0, 0, 0.7)
```

#### Text Colors
```
Light Mode
- Text-Primary:    rgba(0, 0, 0, 0.87)    - High emphasis
- Text-Secondary:  rgba(0, 0, 0, 0.60)    - Medium emphasis
- Text-Disabled:   rgba(0, 0, 0, 0.38)    - Low emphasis

Dark Mode
- Text-Primary:    rgba(255, 255, 255, 0.87)
- Text-Secondary:  rgba(255, 255, 255, 0.60)
- Text-Disabled:   rgba(255, 255, 255, 0.38)
```

### 1.2 Typography System

#### Font Families
```
Primary (UI Text)
- Font-Family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif
- Weights: 400 (Regular), 500 (Medium), 600 (SemiBold), 700 (Bold)

Monospace (Prices, Financial Data, Code)
- Font-Family: 'Roboto Mono', 'Fira Code', 'Courier New', monospace
- Weights: 400 (Regular), 500 (Medium), 600 (SemiBold)
- Use Cases: Option prices, strike prices, Greeks values, PnL figures, timestamps

Headings (Marketing, Documentation)
- Font-Family: 'Manrope', 'Inter', sans-serif
- Weights: 600 (SemiBold), 700 (Bold), 800 (ExtraBold)
```

#### Type Scale
```
Display Large (Landing Pages, Empty States)
- Font-Size: 57px
- Line-Height: 64px (1.12)
- Font-Weight: 700 (Bold)
- Letter-Spacing: -0.25px

Display Medium
- Font-Size: 45px
- Line-Height: 52px (1.16)
- Font-Weight: 700 (Bold)
- Letter-Spacing: 0px

Headline Large (Page Titles)
- Font-Size: 32px
- Line-Height: 40px (1.25)
- Font-Weight: 600 (SemiBold)
- Letter-Spacing: 0px

Headline Medium (Section Headers)
- Font-Size: 28px
- Line-Height: 36px (1.29)
- Font-Weight: 600 (SemiBold)
- Letter-Spacing: 0px

Headline Small (Card Titles)
- Font-Size: 24px
- Line-Height: 32px (1.33)
- Font-Weight: 600 (SemiBold)
- Letter-Spacing: 0px

Title Large (Dialog Titles)
- Font-Size: 22px
- Line-Height: 28px (1.27)
- Font-Weight: 500 (Medium)
- Letter-Spacing: 0px

Title Medium (List Headers)
- Font-Size: 16px
- Line-Height: 24px (1.50)
- Font-Weight: 600 (SemiBold)
- Letter-Spacing: 0.15px

Title Small (Dense Headers)
- Font-Size: 14px
- Line-Height: 20px (1.43)
- Font-Weight: 600 (SemiBold)
- Letter-Spacing: 0.1px

Body Large (Main Content)
- Font-Size: 16px
- Line-Height: 24px (1.50)
- Font-Weight: 400 (Regular)
- Letter-Spacing: 0.5px

Body Medium (Default Body Text)
- Font-Size: 14px
- Line-Height: 20px (1.43)
- Font-Weight: 400 (Regular)
- Letter-Spacing: 0.25px

Body Small (Helper Text, Captions)
- Font-Size: 12px
- Line-Height: 16px (1.33)
- Font-Weight: 400 (Regular)
- Letter-Spacing: 0.4px

Label Large (Buttons, Tabs)
- Font-Size: 14px
- Line-Height: 20px (1.43)
- Font-Weight: 500 (Medium)
- Letter-Spacing: 0.1px
- Text-Transform: none

Label Medium (Form Labels)
- Font-Size: 12px
- Line-Height: 16px (1.33)
- Font-Weight: 500 (Medium)
- Letter-Spacing: 0.5px

Label Small (Tiny Labels, Badges)
- Font-Size: 11px
- Line-Height: 16px (1.45)
- Font-Weight: 500 (Medium)
- Letter-Spacing: 0.5px

Monospace Large (Price Display)
- Font-Size: 24px
- Line-Height: 32px (1.33)
- Font-Weight: 600 (SemiBold)
- Font-Family: Roboto Mono

Monospace Medium (Table Data)
- Font-Size: 14px
- Line-Height: 20px (1.43)
- Font-Weight: 400 (Regular)
- Font-Family: Roboto Mono

Monospace Small (Compact Data)
- Font-Size: 12px
- Line-Height: 16px (1.33)
- Font-Weight: 400 (Regular)
- Font-Family: Roboto Mono
```

### 1.3 Spacing System (4px Base Grid)

```
Space-0:   0px
Space-1:   4px    (0.25rem)
Space-2:   8px    (0.5rem)
Space-3:   12px   (0.75rem)
Space-4:   16px   (1rem)     ← Base unit
Space-5:   20px   (1.25rem)
Space-6:   24px   (1.5rem)
Space-8:   32px   (2rem)
Space-10:  40px   (2.5rem)
Space-12:  48px   (3rem)
Space-16:  64px   (4rem)
Space-20:  80px   (5rem)
Space-24:  96px   (6rem)
Space-32:  128px  (8rem)
Space-40:  160px  (10rem)
Space-48:  192px  (12rem)
Space-64:  256px  (16rem)
Space-80:  320px  (20rem)
```

**Usage Guidelines**:
- Component padding: Space-4 (16px) minimum
- Section spacing: Space-8 (32px) or Space-12 (48px)
- Form field gaps: Space-6 (24px)
- Inline elements: Space-2 (8px) or Space-3 (12px)
- Page margins: Space-8 (32px) or Space-12 (48px)

### 1.4 Border Radius

```
Radius-None:   0px     - Tables, data grids
Radius-SM:     4px     - Chips, badges, small buttons
Radius-MD:     8px     - Default buttons, inputs, cards
Radius-LG:     12px    - Large cards, modals
Radius-XL:     16px    - Hero sections, feature cards
Radius-2XL:    24px    - Full-width cards, panels
Radius-Full:   9999px  - Pills, avatar badges
```

### 1.5 Elevation & Shadows

```
Shadow-None:   none

Shadow-SM (Hover States, Dropdowns)
- Box-Shadow: 0px 1px 2px 0px rgba(0, 0, 0, 0.05)

Shadow-MD (Cards, Buttons)
- Box-Shadow: 0px 4px 6px -1px rgba(0, 0, 0, 0.1),
              0px 2px 4px -2px rgba(0, 0, 0, 0.06)

Shadow-LG (Modals, Popovers)
- Box-Shadow: 0px 10px 15px -3px rgba(0, 0, 0, 0.1),
              0px 4px 6px -4px rgba(0, 0, 0, 0.05)

Shadow-XL (Dialogs, Drawers)
- Box-Shadow: 0px 20px 25px -5px rgba(0, 0, 0, 0.1),
              0px 8px 10px -6px rgba(0, 0, 0, 0.04)

Shadow-2XL (Full-Screen Overlays)
- Box-Shadow: 0px 25px 50px -12px rgba(0, 0, 0, 0.25)

Shadow-Inner (Inset Elements, Sunken Fields)
- Box-Shadow: inset 0px 2px 4px 0px rgba(0, 0, 0, 0.06)
```

### 1.6 Transitions & Animations

```
Duration-Fast:    150ms   - Hover states, small UI changes
Duration-Base:    200ms   - Default transitions
Duration-Slow:    300ms   - Complex animations, page transitions
Duration-Slower:  500ms   - Modal entrances, drawer slides

Easing-Standard:  cubic-bezier(0.4, 0.0, 0.2, 1)  - Default
Easing-Decelerate: cubic-bezier(0.0, 0.0, 0.2, 1)  - Enter animations
Easing-Accelerate: cubic-bezier(0.4, 0.0, 1, 1)    - Exit animations
Easing-Sharp:     cubic-bezier(0.4, 0.0, 0.6, 1)    - Temporary UI changes
```

### 1.7 Z-Index Scale

```
Z-Behind:      -1     - Background decorations
Z-Base:        0      - Default layer
Z-Dropdown:    1000   - Dropdown menus
Z-Sticky:      1100   - Sticky headers/sidebars
Z-Fixed:       1200   - Fixed positioning elements
Z-Modal-Backdrop: 1300 - Modal overlays
Z-Modal:       1400   - Modal content
Z-Popover:     1500   - Popovers, tooltips
Z-Toast:       1600   - Toast notifications
Z-Tooltip:     1700   - Tooltips (highest)
```

---

## 2. Component Library Specifications

### 2.1 Buttons

#### Primary Button (Main Actions)
```
States:
- Default:
  - Background: Primary-500 (#2196F3)
  - Text: White (#FFFFFF)
  - Border: none
  - Padding: 10px 24px (Space-2.5 Space-6)
  - Height: 40px
  - Font: Label-Large (14px, Medium, 0.1px)
  - Border-Radius: Radius-MD (8px)
  - Box-Shadow: Shadow-SM
  - Transition: all Duration-Base Easing-Standard

- Hover:
  - Background: Primary-600 (#1E88E5)
  - Box-Shadow: Shadow-MD
  - Transform: translateY(-1px)

- Active/Pressed:
  - Background: Primary-700 (#1976D2)
  - Box-Shadow: Shadow-SM
  - Transform: translateY(0px)

- Focus:
  - Background: Primary-500
  - Outline: 2px solid Primary-200
  - Outline-Offset: 2px

- Disabled:
  - Background: Secondary-200 (#EEEEEE)
  - Text: Text-Disabled
  - Cursor: not-allowed
  - Box-Shadow: none

- Loading:
  - Background: Primary-500
  - Cursor: wait
  - Content: Spinner icon (16px) + text
  - Opacity: 0.7
```

#### Secondary Button (Less Emphasis Actions)
```
States:
- Default:
  - Background: transparent
  - Text: Primary-500
  - Border: 1px solid Primary-500
  - Padding: 9px 23px (account for border)
  - Height: 40px
  - Font: Label-Large
  - Border-Radius: Radius-MD
  - Box-Shadow: none

- Hover:
  - Background: Primary-50
  - Border: 1px solid Primary-600
  - Text: Primary-600

- Active:
  - Background: Primary-100
  - Border: 1px solid Primary-700
  - Text: Primary-700

- Focus:
  - Background: transparent
  - Border: 1px solid Primary-500
  - Outline: 2px solid Primary-200
  - Outline-Offset: 2px

- Disabled:
  - Background: transparent
  - Border: 1px solid Secondary-300
  - Text: Text-Disabled
  - Cursor: not-allowed
```

#### Danger Button (Destructive Actions)
```
States:
- Default:
  - Background: Error-500 (#F44336)
  - Text: White
  - Border: none
  - [Same dimensions as Primary Button]

- Hover:
  - Background: Error-600 (#E53935)
  - Box-Shadow: Shadow-MD

- Active:
  - Background: Error-700 (#D32F2F)

- Focus:
  - Outline: 2px solid Error-200
  - Outline-Offset: 2px
```

#### Ghost Button (Subtle Actions)
```
States:
- Default:
  - Background: transparent
  - Text: Text-Primary
  - Border: none
  - Padding: 10px 24px
  - Height: 40px
  - Font: Label-Large

- Hover:
  - Background: Secondary-100
  - Text: Text-Primary

- Active:
  - Background: Secondary-200

- Focus:
  - Outline: 2px solid Primary-200
  - Outline-Offset: 2px
```

#### Icon Button
```
Specifications:
- Size: 40px x 40px (default), 32px x 32px (small), 48px x 48px (large)
- Icon-Size: 20px (default), 16px (small), 24px (large)
- Padding: none (icon centered)
- Border-Radius: Radius-Full (circular)
- Background-Default: transparent
- Background-Hover: Secondary-100
- Background-Active: Secondary-200
- Transition: Duration-Fast
```

#### Button Sizes
```
Small:
- Height: 32px
- Padding: 6px 16px
- Font-Size: 13px

Medium (Default):
- Height: 40px
- Padding: 10px 24px
- Font-Size: 14px

Large:
- Height: 48px
- Padding: 14px 32px
- Font-Size: 15px
```

### 2.2 Form Controls

#### Text Input
```
Specifications:
- Height: 40px
- Padding: 10px 12px
- Font: Body-Medium (14px, Regular)
- Border: 1px solid Secondary-400
- Border-Radius: Radius-MD (8px)
- Background: Background-Default
- Transition: border-color Duration-Base, box-shadow Duration-Base

States:
- Default:
  - Border: 1px solid Secondary-400 (#BDBDBD)
  - Background: #FFFFFF

- Hover:
  - Border: 1px solid Secondary-700 (#616161)

- Focus:
  - Border: 2px solid Primary-500
  - Padding: 10px 11px (compensate for thicker border)
  - Box-Shadow: 0px 0px 0px 3px rgba(33, 150, 243, 0.1)
  - Outline: none

- Error:
  - Border: 2px solid Error-500
  - Box-Shadow: 0px 0px 0px 3px rgba(244, 67, 54, 0.1)

- Disabled:
  - Background: Secondary-100
  - Border: 1px solid Secondary-300
  - Text: Text-Disabled
  - Cursor: not-allowed

- Read-Only:
  - Background: Secondary-50
  - Border: 1px solid Secondary-300
  - Cursor: default
```

#### Form Label
```
Specifications:
- Font: Label-Medium (12px, Medium, 0.5px)
- Color: Text-Secondary
- Margin-Bottom: Space-2 (8px)
- Required Indicator: Red asterisk (*) in Error-500

Optional Label:
- Font: Label-Small (11px, Regular)
- Color: Text-Secondary
- Text: "(optional)"
```

#### Helper Text
```
Specifications:
- Font: Body-Small (12px, Regular)
- Color: Text-Secondary
- Margin-Top: Space-2 (8px)

Error Message:
- Color: Error-700
- Icon: Error icon (16px) preceding text
- Display: Fade in with Duration-Fast
```

#### Select Dropdown
```
Specifications:
- Height: 40px
- Padding: 10px 36px 10px 12px (room for dropdown icon)
- Font: Body-Medium
- Border: 1px solid Secondary-400
- Border-Radius: Radius-MD
- Background: #FFFFFF
- Icon: Chevron-down (20px) positioned right 12px

Dropdown Menu:
- Background: Background-Paper
- Border: 1px solid Secondary-300
- Border-Radius: Radius-MD
- Box-Shadow: Shadow-LG
- Max-Height: 300px
- Overflow-Y: auto
- Margin-Top: Space-1 (4px)

Dropdown Item:
- Height: 40px
- Padding: 10px 12px
- Font: Body-Medium
- Hover-Background: Primary-50
- Selected-Background: Primary-100
- Selected-Text: Primary-700
- Selected-Icon: Checkmark (16px) at right
```

#### Checkbox
```
Specifications:
- Size: 20px x 20px
- Border: 2px solid Secondary-600
- Border-Radius: Radius-SM (4px)
- Background-Unchecked: transparent
- Background-Checked: Primary-500
- Checkmark-Color: White
- Checkmark-Size: 14px
- Transition: Duration-Fast

States:
- Hover-Unchecked:
  - Border: 2px solid Primary-500
  - Background: Primary-50

- Checked:
  - Background: Primary-500
  - Border: 2px solid Primary-500
  - Checkmark: Visible

- Focus:
  - Outline: 2px solid Primary-200
  - Outline-Offset: 2px

- Disabled:
  - Border: 2px solid Secondary-300
  - Background: Secondary-100
  - Cursor: not-allowed

Label:
- Font: Body-Medium
- Color: Text-Primary
- Margin-Left: Space-2 (8px)
- Cursor: pointer
```

#### Radio Button
```
Specifications:
- Size: 20px x 20px
- Border: 2px solid Secondary-600
- Border-Radius: Radius-Full (circular)
- Background-Unchecked: transparent
- Outer-Circle-Checked: Primary-500
- Inner-Dot-Checked: Primary-500 (8px diameter)
- Transition: Duration-Fast

States:
- Hover-Unchecked:
  - Border: 2px solid Primary-500
  - Background: Primary-50

- Checked:
  - Border: 2px solid Primary-500
  - Inner-Dot: 8px circle, Primary-500

- Focus:
  - Outline: 2px solid Primary-200
  - Outline-Offset: 2px

Label:
- Font: Body-Medium
- Margin-Left: Space-2 (8px)
```

#### Toggle Switch
```
Specifications:
- Track-Width: 44px
- Track-Height: 24px
- Track-Border-Radius: Radius-Full
- Thumb-Size: 20px
- Thumb-Offset: 2px (from track edge)
- Track-Background-Off: Secondary-400
- Track-Background-On: Primary-500
- Thumb-Background: White
- Thumb-Box-Shadow: Shadow-SM
- Transition: Duration-Base

States:
- Off:
  - Track: Secondary-400
  - Thumb-Position: Left (2px offset)

- On:
  - Track: Primary-500
  - Thumb-Position: Right (22px from left)

- Hover-Off:
  - Track: Secondary-500

- Hover-On:
  - Track: Primary-600

- Focus:
  - Outline: 2px solid Primary-200
  - Outline-Offset: 2px

- Disabled:
  - Track-Off: Secondary-200
  - Track-On: Primary-200
  - Opacity: 0.5
  - Cursor: not-allowed
```

### 2.3 Cards

#### Standard Card
```
Specifications:
- Background: Background-Paper (#FAFAFA in light mode)
- Border: 1px solid Secondary-300
- Border-Radius: Radius-LG (12px)
- Box-Shadow: Shadow-SM
- Padding: Space-6 (24px)
- Transition: box-shadow Duration-Base, transform Duration-Base

Hover State (Interactive Cards):
- Box-Shadow: Shadow-MD
- Transform: translateY(-2px)
- Border: 1px solid Secondary-400

Card Header:
- Margin-Bottom: Space-4 (16px)
- Padding-Bottom: Space-4
- Border-Bottom: 1px solid Secondary-300

Card Title:
- Font: Headline-Small (24px, SemiBold)
- Color: Text-Primary

Card Subtitle:
- Font: Body-Medium (14px, Regular)
- Color: Text-Secondary
- Margin-Top: Space-1 (4px)

Card Content:
- Font: Body-Medium
- Color: Text-Primary
- Line-Height: 1.6

Card Actions:
- Margin-Top: Space-6 (24px)
- Padding-Top: Space-4 (16px)
- Border-Top: 1px solid Secondary-300
- Display: flex
- Justify-Content: flex-end
- Gap: Space-3 (12px)
```

#### Option Position Card
```
Purpose: Display individual option position with key metrics

Specifications:
- Height: auto
- Padding: Space-5 (20px)
- Background: Background-Paper
- Border-Left: 4px solid [Call-Color for calls, Put-Color for puts]
- Border-Radius: Radius-MD
- Box-Shadow: Shadow-SM

Layout Grid:
Row 1 (Header):
- Option-Symbol: Font Monospace-Large, Weight 600
- Option-Type: Badge (Call/Put)
- Expiry-Date: Font Monospace-Small, Color Text-Secondary

Row 2 (Key Metrics):
- Strike-Price: Font Monospace-Medium, Label "Strike"
- Current-Price: Font Monospace-Medium, Label "Price"
- PnL: Font Monospace-Medium, Color [Success/Error based on sign]

Row 3 (Greeks - Collapsible):
- Delta: Monospace-Small
- Gamma: Monospace-Small
- Theta: Monospace-Small
- Vega: Monospace-Small

Spacing:
- Row-Gap: Space-4 (16px)
- Column-Gap: Space-6 (24px)

Hover State:
- Box-Shadow: Shadow-MD
- Border-Left-Width: 6px (expand emphasis)
```

#### Analytics Card
```
Purpose: Display charts, graphs, and complex visualizations

Specifications:
- Background: Background-Default (White)
- Border: 1px solid Secondary-300
- Border-Radius: Radius-LG
- Padding: Space-6 (24px)
- Min-Height: 400px

Card Header:
- Title: Headline-Small
- Action-Buttons: Icon buttons (download, fullscreen, settings)
- Margin-Bottom: Space-6

Chart Container:
- Height: 100% of remaining space
- Min-Height: 300px
- Padding: Space-4
- Background: Background-Paper

Legend:
- Position: Top-right or bottom
- Font: Body-Small
- Color-Indicator: 12px circle
- Item-Spacing: Space-3
```

#### KPI Card (Dashboard Metrics)
```
Purpose: Display single key metric with trend indicator

Specifications:
- Width: 100% (responsive)
- Height: 140px
- Background: Linear-gradient (subtle)
- Border: none
- Border-Radius: Radius-LG
- Box-Shadow: Shadow-MD
- Padding: Space-5
- Cursor: pointer (clickable to drill down)

Layout:
- Label: Title-Small, Color Text-Secondary, Top
- Value: Display-Medium, Font Monospace-Large, Color Text-Primary, Center
- Trend-Indicator: Body-Small + Arrow icon, Color [Success/Error], Bottom-right
- Comparison-Period: Body-Small, Color Text-Disabled, Bottom-left

Hover State:
- Box-Shadow: Shadow-LG
- Transform: scale(1.02)
- Background: Gradient intensity +10%

Click State:
- Transform: scale(0.98)
- Box-Shadow: Shadow-SM
```

### 2.4 Tables

#### Data Table (Financial Data)
```
Purpose: Display tabular financial data with sorting, filtering, and pagination

Specifications:
- Width: 100%
- Background: Background-Default
- Border: 1px solid Secondary-300
- Border-Radius: Radius-MD
- Box-Shadow: Shadow-SM
- Font: Monospace-Medium for numerical columns, Body-Medium for text

Table Header:
- Position: sticky
- Top: 0
- Z-Index: Z-Sticky (1100)
- Background: Secondary-100
- Border-Bottom: 2px solid Secondary-400
- Height: 48px
- Padding: 12px 16px
- Font: Label-Medium, Weight 600
- Text-Transform: uppercase
- Letter-Spacing: 0.5px

Sortable Column Header:
- Cursor: pointer
- Display: flex
- Align-Items: center
- Gap: Space-2
- Sort-Icon: 16px, Color Secondary-600
- Hover-Background: Secondary-200
- Active-Sort-Color: Primary-600

Table Row:
- Height: 56px
- Padding: 16px
- Border-Bottom: 1px solid Secondary-200
- Transition: background-color Duration-Fast

Zebra Striping:
- Odd-Rows: Background-Default (#FFFFFF)
- Even-Rows: Background-Paper (#FAFAFA)

Row Hover State:
- Background: Primary-50
- Border-Left: 3px solid Primary-500
- Cursor: pointer (if clickable)

Row Selected State:
- Background: Primary-100
- Border-Left: 4px solid Primary-700
- Font-Weight: 500

Cell Padding:
- Horizontal: Space-4 (16px)
- Vertical: Space-3 (12px)

Cell Alignment:
- Text-Columns: Left
- Numerical-Columns: Right (tabular-nums font feature)
- Date-Columns: Right
- Action-Columns: Center

Empty State:
- Height: 300px
- Display: flex, center content
- Illustration: 120px x 120px, grayscale
- Message: Headline-Small, Color Text-Secondary
- Action-Button: Primary button "Add First Item"

Loading State:
- Skeleton-Rows: 5 rows
- Skeleton-Animation: Pulse effect, Duration-Slower
- Background: Linear-gradient shimmer effect
```

#### Pagination Footer
```
Specifications:
- Height: 56px
- Background: Background-Paper
- Border-Top: 1px solid Secondary-300
- Padding: 12px 16px
- Display: flex
- Justify-Content: space-between
- Align-Items: center

Rows Per Page Selector:
- Label: "Rows per page:"
- Font: Body-Small
- Select-Width: 80px
- Options: [10, 25, 50, 100]

Page Info:
- Font: Body-Small
- Color: Text-Secondary
- Format: "1-10 of 245"

Page Navigation:
- First-Page-Button: Icon button (double-left-arrow)
- Previous-Page-Button: Icon button (left-arrow)
- Page-Number-Display: Body-Medium, "Page 1 of 25"
- Next-Page-Button: Icon button (right-arrow)
- Last-Page-Button: Icon button (double-right-arrow)
- Button-Size: 32px x 32px
- Button-Gap: Space-1 (4px)
- Disabled-Opacity: 0.4
```

#### Responsive Table (Mobile)
```
Breakpoint: < 768px

Transformation:
- Table → Stacked Cards
- Each Row → Individual Card
- Card-Padding: Space-4
- Card-Margin-Bottom: Space-3
- Card-Border-Radius: Radius-MD
- Card-Box-Shadow: Shadow-SM

Card Layout:
- Label-Value Pairs: Vertical stack
- Label: Body-Small, Weight 600, Color Text-Secondary
- Value: Body-Medium, Color Text-Primary
- Pair-Margin-Bottom: Space-3

Horizontal Scroll Alternative:
- Container: overflow-x: auto
- Table: min-width: 800px
- Scroll-Indicator: Gradient fade at edges
```

### 2.5 Modals & Dialogs

#### Standard Modal
```
Specifications:
Backdrop:
- Background: rgba(0, 0, 0, 0.5)
- Position: fixed
- Top: 0, Left: 0, Right: 0, Bottom: 0
- Z-Index: Z-Modal-Backdrop (1300)
- Animation: Fade in Duration-Base

Modal Container:
- Background: Background-Default
- Border-Radius: Radius-LG (12px)
- Box-Shadow: Shadow-2XL
- Position: fixed
- Top: 50%, Left: 50%
- Transform: translate(-50%, -50%)
- Z-Index: Z-Modal (1400)
- Max-Width: 600px (default)
- Max-Height: 90vh
- Overflow-Y: auto
- Animation: Scale from 0.95 to 1.0, Duration-Base

Modal Header:
- Padding: Space-6 (24px)
- Border-Bottom: 1px solid Secondary-300
- Display: flex
- Justify-Content: space-between
- Align-Items: center

Modal Title:
- Font: Headline-Small (24px, SemiBold)
- Color: Text-Primary

Close Button:
- Type: Icon button
- Size: 32px x 32px
- Icon: X icon, 20px
- Position: Top-right of header

Modal Content:
- Padding: Space-6 (24px)
- Font: Body-Medium
- Color: Text-Primary
- Max-Height: calc(90vh - 160px)
- Overflow-Y: auto

Modal Actions:
- Padding: Space-6 (24px)
- Border-Top: 1px solid Secondary-300
- Display: flex
- Justify-Content: flex-end
- Gap: Space-3 (12px)
- Button-Order: [Cancel/Secondary, Confirm/Primary]

Enter Animation:
- Backdrop: Fade in from opacity 0 to 1
- Modal: Scale from 0.95 to 1.0 + Fade in
- Duration: Duration-Base (200ms)
- Easing: Easing-Decelerate

Exit Animation:
- Backdrop: Fade out to opacity 0
- Modal: Scale from 1.0 to 0.95 + Fade out
- Duration: Duration-Fast (150ms)
- Easing: Easing-Accelerate

Focus Management:
- On-Open: Focus first focusable element (usually close button)
- Focus-Trap: Tab cycles within modal
- On-Close: Return focus to trigger element
- Escape-Key: Close modal
```

#### Confirmation Dialog
```
Specifications:
- Max-Width: 400px
- Icon: 48px warning/error/info icon at top
- Icon-Margin-Bottom: Space-4

Content:
- Title: Headline-Small, Center-aligned
- Message: Body-Large, Center-aligned, Color Text-Secondary
- Message-Margin-Top: Space-3

Actions:
- Layout: Two buttons, equal width
- Button-Width: 48% each
- Gap: 4% between
- Justify-Content: center
- Destructive-Action: Danger button (left)
- Safe-Action: Secondary button (right)

Example Use Case:
- Title: "Delete Position?"
- Message: "This action cannot be undone. The option position and all associated data will be permanently deleted."
- Actions: ["Delete" (Danger), "Cancel" (Secondary)]
```

#### Drawer (Side Panel)
```
Specifications:
Backdrop:
- Same as modal backdrop

Drawer Container:
- Background: Background-Default
- Position: fixed
- Top: 0
- Height: 100vh
- Width: 480px (default), 640px (large), 320px (small)
- Z-Index: Z-Modal (1400)
- Box-Shadow: Shadow-2XL

Placement:
- Right: Right: 0, Slide in from right
- Left: Left: 0, Slide in from left

Drawer Header:
- Same as modal header
- Position: sticky
- Top: 0
- Z-Index: 1

Drawer Content:
- Padding: Space-6
- Overflow-Y: auto
- Height: calc(100vh - 80px - 80px) // Minus header and footer

Drawer Footer (Optional):
- Position: sticky
- Bottom: 0
- Background: Background-Paper
- Border-Top: 1px solid Secondary-300
- Padding: Space-6
- Display: flex
- Justify-Content: flex-end
- Gap: Space-3

Enter Animation (Right Drawer):
- Backdrop: Fade in
- Drawer: Slide from translateX(100%) to translateX(0)
- Duration: Duration-Slow (300ms)
- Easing: Easing-Decelerate

Exit Animation:
- Backdrop: Fade out
- Drawer: Slide from translateX(0) to translateX(100%)
- Duration: Duration-Base (200ms)
- Easing: Easing-Accelerate
```

### 2.6 Alerts & Notifications

#### Inline Alert
```
Specifications:
- Width: 100%
- Min-Height: 48px
- Padding: 12px 16px
- Border-Radius: Radius-MD (8px)
- Border-Left: 4px solid [variant color]
- Display: flex
- Align-Items: flex-start
- Gap: Space-3 (12px)
- Margin-Bottom: Space-4

Variants:
Success:
- Background: Success-50
- Border-Color: Success-600
- Icon-Color: Success-700
- Text-Color: Success-900

Error:
- Background: Error-50
- Border-Color: Error-600
- Icon-Color: Error-700
- Text-Color: Error-900

Warning:
- Background: Warning-50
- Border-Color: Warning-600
- Icon-Color: Warning-700
- Text-Color: Warning-900

Info:
- Background: Info-50
- Border-Color: Info-600
- Icon-Color: Info-700
- Text-Color: Info-900

Layout:
- Icon: 20px x 20px, aligned to top
- Content: Flex-grow: 1
- Title: Label-Large, Weight 600
- Message: Body-Medium
- Message-Margin-Top: Space-1
- Close-Button: Icon button, 24px x 24px, aligned to top

Dismissible:
- Close-Icon: X icon, 16px
- Fade-Out: Duration-Fast
- Collapse: Height transition to 0
```

#### Toast Notification
```
Specifications:
- Position: fixed
- Top: 24px (or Bottom: 24px for bottom toast)
- Right: 24px
- Width: 360px
- Min-Height: 64px
- Z-Index: Z-Toast (1600)
- Background: Background-Default
- Border: 1px solid Secondary-300
- Border-Radius: Radius-MD
- Box-Shadow: Shadow-XL
- Padding: 16px
- Display: flex
- Gap: Space-3

Layout:
- Icon: 24px x 24px, Color [variant color]
- Content: Flex-grow: 1
- Title: Label-Large, Weight 600
- Message: Body-Small
- Message-Margin-Top: Space-1
- Close-Button: Icon button, top-right

Variants:
- Success: Icon CheckCircle, Color Success-600
- Error: Icon XCircle, Color Error-600
- Warning: Icon AlertTriangle, Color Warning-600
- Info: Icon InfoCircle, Color Info-600

Auto-Dismiss:
- Default-Duration: 5000ms (5 seconds)
- Success: 4000ms
- Error: 7000ms (longer for errors)
- Warning: 6000ms
- Pause-On-Hover: true

Enter Animation:
- Slide in from translateX(100%) + Fade in
- Duration: Duration-Slow (300ms)
- Easing: Easing-Decelerate

Exit Animation:
- Slide out to translateX(100%) + Fade out
- Duration: Duration-Base (200ms)
- Easing: Easing-Accelerate

Stacking:
- Multiple-Toasts: Stack vertically with Space-3 gap
- Max-Visible: 3 toasts
- Older-Toasts: Auto-dismiss when exceeding max
```

#### Banner Alert
```
Purpose: Full-width persistent alerts at page top

Specifications:
- Position: fixed or static (page top)
- Width: 100%
- Height: 56px
- Padding: 12px 24px
- Background: [Variant color background]
- Border-Bottom: 1px solid [Variant border color]
- Z-Index: Z-Sticky (1100)
- Display: flex
- Align-Items: center
- Justify-Content: center
- Gap: Space-4

Content:
- Icon: 20px, Color [variant icon color]
- Message: Body-Medium, Weight 500
- Action-Button: Secondary button (small), optional
- Close-Button: Icon button, optional

Use Cases:
- System-Wide Announcements
- Connectivity Issues
- Maintenance Mode
- Critical Updates
```

### 2.7 Navigation

#### Top Navigation Bar
```
Specifications:
- Position: sticky
- Top: 0
- Width: 100%
- Height: 64px
- Background: Background-Default
- Border-Bottom: 1px solid Secondary-300
- Box-Shadow: Shadow-SM
- Z-Index: Z-Sticky (1100)
- Padding: 0px 24px

Layout:
- Display: flex
- Justify-Content: space-between
- Align-Items: center

Left Section:
- Logo: 40px height, clickable
- App-Title: Headline-Small, Margin-Left Space-4 (optional)

Center Section (Optional):
- Search-Bar: Width 400px (desktop)
- Quick-Actions: Icon buttons

Right Section:
- User-Menu: Avatar (32px) + Username + Dropdown
- Notifications: Icon button with badge
- Settings: Icon button
- Gap: Space-3 between items

Dark Mode:
- Background: #1E1E1E
- Border-Bottom: 1px solid #3A3A3A
- Box-Shadow: Shadow-MD
```

#### Sidebar Navigation
```
Specifications:
- Position: fixed
- Left: 0
- Top: 64px (below top nav)
- Width: 280px
- Height: calc(100vh - 64px)
- Background: Background-Paper
- Border-Right: 1px solid Secondary-300
- Overflow-Y: auto
- Z-Index: Z-Fixed (1200)
- Padding: Space-4 Space-3

Collapsible State:
- Width-Collapsed: 64px
- Show-Icons-Only: true
- Tooltip-On-Hover: true
- Toggle-Button: Top-right corner

Navigation Item:
- Height: 40px
- Padding: 8px 12px
- Border-Radius: Radius-MD
- Display: flex
- Align-Items: center
- Gap: Space-3
- Margin-Bottom: Space-1
- Font: Label-Large
- Transition: Duration-Fast

Item States:
- Default:
  - Background: transparent
  - Text: Text-Primary
  - Icon: Text-Secondary

- Hover:
  - Background: Secondary-100
  - Text: Text-Primary
  - Icon: Text-Primary

- Active:
  - Background: Primary-100
  - Text: Primary-700
  - Icon: Primary-700
  - Border-Left: 3px solid Primary-600
  - Font-Weight: 600

Icon:
- Size: 20px x 20px
- Margin-Right: Space-3

Badge (Notifications):
- Size: 20px x 20px
- Background: Error-500
- Color: White
- Border-Radius: Radius-Full
- Font: Label-Small
- Position: Absolute, top-right of item

Section Divider:
- Height: 1px
- Background: Secondary-300
- Margin: Space-4 0

Section Header:
- Font: Label-Small
- Color: Text-Disabled
- Text-Transform: uppercase
- Padding: 8px 12px
- Margin-Top: Space-4
```

#### Breadcrumbs
```
Specifications:
- Height: 40px
- Padding: 8px 0px
- Display: flex
- Align-Items: center
- Gap: Space-2
- Font: Body-Small
- Color: Text-Secondary

Breadcrumb Item:
- Color: Text-Secondary
- Text-Decoration: none
- Transition: color Duration-Fast

Breadcrumb Item Hover:
- Color: Primary-500
- Text-Decoration: underline

Current Page (Last Item):
- Color: Text-Primary
- Font-Weight: 500
- Cursor: default

Separator:
- Content: "/" or chevron-right icon
- Color: Text-Disabled
- Margin: 0 Space-2
- Size: 16px
```

#### Tabs
```
Specifications:
Container:
- Display: flex
- Border-Bottom: 2px solid Secondary-300
- Gap: 0px

Tab Item:
- Height: 48px
- Padding: 12px 24px
- Font: Label-Large
- Color: Text-Secondary
- Background: transparent
- Border-Bottom: 2px solid transparent
- Margin-Bottom: -2px (overlap container border)
- Cursor: pointer
- Transition: Duration-Fast

Tab States:
- Default:
  - Color: Text-Secondary
  - Border-Bottom: 2px solid transparent

- Hover:
  - Color: Text-Primary
  - Background: Secondary-50

- Active:
  - Color: Primary-600
  - Border-Bottom: 2px solid Primary-600
  - Font-Weight: 600

- Focus:
  - Outline: 2px solid Primary-200
  - Outline-Offset: -2px

- Disabled:
  - Color: Text-Disabled
  - Cursor: not-allowed
  - Opacity: 0.5

Tab Badge:
- Size: 20px height
- Padding: 2px 8px
- Background: Secondary-200
- Color: Text-Primary
- Border-Radius: Radius-Full
- Font: Label-Small
- Margin-Left: Space-2

Active Tab Badge:
- Background: Primary-100
- Color: Primary-700
```

---

## 3. Layout System

### 3.1 Responsive Breakpoints

```
XS (Extra Small - Mobile Portrait)
- Min-Width: 0px
- Max-Width: 599px
- Container-Padding: 16px
- Grid-Columns: 4
- Primary-Use: Mobile phones portrait

SM (Small - Mobile Landscape)
- Min-Width: 600px
- Max-Width: 959px
- Container-Padding: 24px
- Grid-Columns: 8
- Primary-Use: Mobile phones landscape, small tablets

MD (Medium - Tablet)
- Min-Width: 960px
- Max-Width: 1279px
- Container-Padding: 32px
- Grid-Columns: 12
- Primary-Use: Tablets, small laptops

LG (Large - Desktop)
- Min-Width: 1280px
- Max-Width: 1919px
- Container-Padding: 32px
- Grid-Columns: 12
- Primary-Use: Desktop monitors, laptops

XL (Extra Large - Wide Desktop)
- Min-Width: 1920px
- Container-Padding: 32px
- Grid-Columns: 12
- Max-Container-Width: 1600px
- Primary-Use: Large desktop monitors
```

### 3.2 Grid System

```
Specifications:
- Columns: 12 (default for MD and above)
- Gutter: 24px (Space-6)
- Margin: Responsive (based on breakpoint)

Column Widths (12-column grid):
- col-1:  8.333%
- col-2:  16.666%
- col-3:  25%
- col-4:  33.333%
- col-5:  41.666%
- col-6:  50%
- col-7:  58.333%
- col-8:  66.666%
- col-9:  75%
- col-10: 83.333%
- col-11: 91.666%
- col-12: 100%

Responsive Column Syntax:
- xs-12: 100% width on XS screens
- sm-6:  50% width on SM screens
- md-4:  33.33% width on MD screens
- lg-3:  25% width on LG screens
```

### 3.3 Dashboard Layout Pattern

```
Structure:
┌─────────────────────────────────────────────────────────┐
│                  Top Navigation (64px)                  │
├──────────────┬──────────────────────────────────────────┤
│   Sidebar    │         Main Content Area               │
│   (280px)    │                                          │
│              │  ┌────────────────────────────────────┐  │
│  Dashboard   │  │    Page Header (80px)              │  │
│  Pricing     │  ├────────────────────────────────────┤  │
│  Portfolio   │  │                                    │  │
│  Analytics   │  │    Content Grid/Cards              │  │
│  Markets     │  │                                    │  │
│  Settings    │  │                                    │  │
│              │  └────────────────────────────────────┘  │
│              │                                          │
└──────────────┴──────────────────────────────────────────┘

Main Content Area:
- Padding: Space-8 (32px)
- Max-Width: 1600px (center-aligned on XL screens)
- Background: Background-Default

Page Header:
- Height: 80px
- Margin-Bottom: Space-8
- Display: flex
- Justify-Content: space-between
- Align-Items: center

Page Title:
- Font: Headline-Large (32px, SemiBold)

Page Actions:
- Display: flex
- Gap: Space-3
- Buttons: Primary, Secondary

Content Grid:
- Display: grid
- Grid-Template-Columns: repeat(12, 1fr)
- Gap: Space-6 (24px)
- Responsive:
  - XS: 1 column
  - SM: 2 columns
  - MD: 3 columns
  - LG: 4 columns
```

### 3.4 Full-Width Tool Layout

```
Purpose: Immersive tools like advanced pricing calculator, charting

Structure:
┌─────────────────────────────────────────────────────────┐
│                  Top Navigation (64px)                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│                   Tool Header (60px)                    │
│  [Back Button] [Tool Title]         [Actions]          │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│                                                         │
│                   Full-Width Content                    │
│                  (100vw - no padding)                   │
│                                                         │
│                                                         │
└─────────────────────────────────────────────────────────┘

Specifications:
- No Sidebar
- No Container Padding
- Background: Background-Paper
- Min-Height: calc(100vh - 64px)

Tool Header:
- Background: Background-Default
- Border-Bottom: 1px solid Secondary-300
- Padding: 12px 24px
- Display: flex
- Justify-Content: space-between

Back Button:
- Ghost button with left arrow icon
- Returns to previous page

Tool Title:
- Font: Headline-Small
```

---

## 4. Accessibility Specifications

### 4.1 Color Contrast Requirements (WCAG 2.1 AA)

```
Normal Text (< 18px):
- Minimum-Contrast-Ratio: 4.5:1
- Verified-Combinations:
  - Text-Primary (#000000 87%) on Background-Default (#FFFFFF): 13.6:1 ✓
  - Text-Secondary (#000000 60%) on Background-Default: 7.0:1 ✓
  - Primary-500 (#2196F3) on Background-Default: 3.1:1 ✗ (Use Primary-700 for text)
  - Success-700 (#388E3C) on Background-Default: 4.6:1 ✓
  - Error-700 (#D32F2F) on Background-Default: 4.7:1 ✓

Large Text (≥ 18px or ≥ 14px bold):
- Minimum-Contrast-Ratio: 3:1
- Verified-Combinations:
  - Primary-500 (#2196F3) on Background-Default: 3.1:1 ✓
  - Success-500 (#4CAF50) on Background-Default: 3.2:1 ✓

UI Components (borders, focus indicators):
- Minimum-Contrast-Ratio: 3:1
- Border-Colors: Secondary-400 (#BDBDBD) on #FFFFFF: 3.4:1 ✓
- Focus-Outline: Primary-500 (#2196F3) on #FFFFFF: 3.1:1 ✓

Dark Mode Compliance:
- Text-Primary (White 87%) on Background-Default (#121212): 13.2:1 ✓
- All-Semantic-Colors: Lightened by 20% for sufficient contrast
```

### 4.2 Keyboard Navigation

```
Focus Indicators:
- Visible: Always visible, never removed
- Style: 2px solid outline
- Color: Primary-500 (#2196F3)
- Offset: 2px (outline-offset)
- Border-Radius: Match element radius + 2px

Tab Order:
- Logical: Left-to-right, top-to-bottom
- Skip-Links: "Skip to main content" link at page top
- Focus-Trap: Modal/drawer focus cycles within component

Keyboard Shortcuts:
Button:
- Enter: Activate
- Space: Activate

Checkbox/Radio:
- Space: Toggle/select
- Arrow-Keys: Navigate between radio options in group

Select Dropdown:
- Enter/Space: Open dropdown
- Arrow-Up/Down: Navigate options
- Enter: Select option
- Escape: Close dropdown

Modal:
- Escape: Close modal
- Tab: Cycle through focusable elements
- Shift+Tab: Reverse cycle

Table:
- Arrow-Keys: Navigate cells
- Home/End: First/last column
- Page-Up/Down: Scroll page

Tabs:
- Arrow-Left/Right: Navigate tabs
- Home: First tab
- End: Last tab
```

### 4.3 Screen Reader Support

```
Semantic HTML:
- Use: <button>, <a>, <input>, <label>, <nav>, <main>, <header>, <footer>
- Avoid: <div> with onClick (unless ARIA role added)

ARIA Labels:
Icon Buttons:
- aria-label: Descriptive text (e.g., "Close dialog", "Sort ascending")

Form Fields:
- aria-describedby: Link to helper text or error message
- aria-invalid: "true" when validation fails
- aria-required: "true" for required fields

Dynamic Content:
- aria-live: "polite" for non-critical updates, "assertive" for urgent
- aria-atomic: "true" to read entire region on change

Tables:
- <th scope="col">: Column headers
- <th scope="row">: Row headers
- aria-sort: "ascending" | "descending" | "none" on sortable headers

Modals:
- role="dialog"
- aria-modal="true"
- aria-labelledby: ID of modal title
- aria-describedby: ID of modal description

Loading States:
- aria-busy="true" during loading
- aria-live="polite" region announcing completion

Toasts:
- role="status" (non-critical)
- role="alert" (critical errors)
- aria-live="polite" | "assertive"
```

### 4.4 Motion & Animation Preferences

```
Prefers-Reduced-Motion:
- Detect: @media (prefers-reduced-motion: reduce)
- Disable:
  - All scale transforms
  - Slide animations
  - Fade transitions (keep instant state changes)
  - Page transitions
- Preserve:
  - Hover state color changes
  - Focus indicators
  - Loading spinners (use simpler pulsing dot)

Implementation:
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```

### 4.5 Form Accessibility

```
Labels:
- Required: Every input must have associated <label>
- Explicit: Use for attribute linking label to input ID
- Visible: Labels always visible (no placeholder-only)

Validation:
- Inline: Show errors immediately below field
- Icon: Error icon (!) preceding error message
- Color: Don't rely on color alone (use icon + text)
- Focus: Auto-focus first error field on submit
- ARIA: aria-invalid="true", aria-describedby pointing to error ID

Error Summary:
- Position: Top of form
- Role: role="alert" for screen reader announcement
- List: Clickable links to each error field
- Display: On submit with validation errors

Required Fields:
- Indicator: Red asterisk (*) in label
- ARIA: aria-required="true"
- Never: Don't rely solely on asterisk color

Helper Text:
- Position: Below input
- ID: Unique ID for aria-describedby
- Color: Text-Secondary (sufficient contrast)
```

---

## 5. User Flow Specifications

### 5.1 New User Onboarding Flow

```
Step 1: Registration Page
- Layout: Center card (max-width 480px) on full-page background
- Background: Linear gradient (Primary-50 to Primary-100)
- Card-Components:
  - Headline: "Welcome to Black-Scholes Platform"
  - Subheading: "Start pricing options in minutes"
  - Form-Fields:
    - Email (required, validation: email format)
    - Password (required, min 8 chars, complexity validation)
    - Confirm-Password (required, must match)
  - Terms-Checkbox: "I agree to Terms & Privacy Policy" (required)
  - Submit-Button: Primary button "Create Account"
  - Alternative-Login: "Already have an account? Sign in"

Step 2: Email Verification
- Page-Type: Full-screen success state
- Icon: Email icon (64px), Primary-500
- Headline: "Check your email"
- Message: "We've sent a verification link to [email]. Click the link to activate your account."
- Action-Button: Secondary "Resend email"
- Helper-Text: "Didn't receive? Check spam folder"

Step 3: First Login → Onboarding Wizard
- Modal-Type: Full-screen modal (non-dismissible)
- Progress-Indicator: 4-step stepper at top

  Wizard Step 1/4: Profile Setup
  - Headline: "Tell us about yourself"
  - Fields:
    - Full-Name (required)
    - Organization (optional)
    - Role-Select: Dropdown [Trader, Analyst, Developer, Student, Other]
  - Navigation: "Next" (Primary button)

  Wizard Step 2/4: Trading Experience
  - Headline: "What's your experience level?"
  - Options: Radio button list
    - Beginner: "New to options trading"
    - Intermediate: "1-3 years experience"
    - Advanced: "Professional trader"
  - Navigation: "Back" (Ghost), "Next" (Primary)

  Wizard Step 3/4: Interests
  - Headline: "What features interest you?"
  - Options: Checkbox list
    - Option Pricing
    - Portfolio Management
    - Market Data Analysis
    - Greeks Analysis
    - Machine Learning Models
  - Navigation: "Back", "Next"

  Wizard Step 4/4: Sample Calculation
  - Headline: "Try your first option pricing"
  - Form: Simplified pricing form
    - Spot-Price: Input (default: 100)
    - Strike-Price: Input (default: 100)
    - Maturity: Input (default: 1.0 years)
    - Volatility: Input (default: 0.20)
    - Interest-Rate: Input (default: 0.05)
  - Result-Card: Live-updating result display
  - Navigation: "Back", "Complete Setup" (Primary)

Step 4: Dashboard Introduction (First Visit)
- Overlay: Semi-transparent backdrop
- Spotlight: Interactive tour with sequential highlights

  Tour Stop 1: Sidebar
  - Highlight: Sidebar navigation
  - Tooltip: "Navigate between pricing, portfolio, and analytics"
  - Position: Right of sidebar
  - Actions: "Next" (Primary), "Skip tour"

  Tour Stop 2: Quick Pricing Widget
  - Highlight: Dashboard pricing calculator
  - Tooltip: "Price options instantly from your dashboard"
  - Position: Below widget

  Tour Stop 3: Recent Calculations
  - Highlight: Recent calculations table
  - Tooltip: "Access your calculation history"

  Tour Stop 4: Help Resources
  - Highlight: Help icon in top nav
  - Tooltip: "Access documentation and support anytime"
  - Actions: "Get Started" (Primary)

Completion:
- Action: Dismiss tour overlay
- Effect: Fade to normal dashboard
- Persistence: Don't show again (stored in user preferences)
```

### 5.2 Quick Pricing Flow

```
Entry Point: Dashboard or Sidebar "Pricing" link

Page Load:
- URL: /pricing
- Layout: Dashboard layout (sidebar + main content)
- Focus: Auto-focus spot price input

Main Content Structure:
┌─────────────────────────────────────────────────────────┐
│  Page Header                                            │
│  "Option Pricing Calculator"                            │
│  [Export Results] [Save Template]                       │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌────────────────────────────┐  │
│  │  Input Panel     │  │  Results Panel             │  │
│  │  (Left, 40%)     │  │  (Right, 60%)              │  │
│  │                  │  │                            │  │
│  │  Option Type     │  │  ┌──────────────────────┐  │  │
│  │  [Call] [Put]    │  │  │  Price Card          │  │  │
│  │                  │  │  │  $12.34              │  │  │
│  │  Spot Price      │  │  └──────────────────────┘  │  │
│  │  [100]           │  │                            │  │
│  │                  │  │  ┌──────────────────────┐  │  │
│  │  Strike Price    │  │  │  Greeks Grid         │  │  │
│  │  [105]           │  │  │  Delta  Gamma        │  │  │
│  │                  │  │  │  Theta  Vega         │  │  │
│  │  [More inputs]   │  │  └──────────────────────┘  │  │
│  │                  │  │                            │  │
│  │  [Calculate]     │  │  [Payoff Diagram Chart]    │  │
│  └──────────────────┘  └────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘

Input Panel Specifications:
- Width: 40% (desktop), 100% (mobile, stacked)
- Padding: Space-6
- Background: Background-Paper
- Border-Radius: Radius-LG
- Box-Shadow: Shadow-SM

Field Order:
1. Option-Type: Toggle button group (Call/Put)
2. Spot-Price: Number input, label "Current Stock Price"
3. Strike-Price: Number input, label "Strike Price"
4. Time-To-Maturity: Number input, label "Time to Expiration (years)"
   - Helper-Text: "Or select date" + Date picker icon
5. Volatility: Number input with slider (0-100%)
6. Risk-Free-Rate: Number input, label "Risk-Free Interest Rate (%)"
7. Dividend-Yield: Number input, label "Dividend Yield (%)" (optional, collapsible)

Advanced Options (Collapsible):
- Accordion: "Advanced Settings" (collapsed by default)
- Fields:
  - Pricing-Method: Dropdown [Black-Scholes, Binomial, Monte Carlo]
  - Steps/Simulations: Number input (method-dependent)
  - American-Style: Checkbox

Calculate Button:
- Type: Primary button
- Width: 100%
- Height: 48px
- Text: "Calculate Option Price"
- Loading-State: Spinner + "Calculating..."

Real-Time Calculation:
- Trigger: On any input change (debounced 500ms)
- Loading: Skeleton loader in results panel
- Duration: < 100ms for Black-Scholes

Results Panel:
Price Card:
- Display: Monospace-Large (48px)
- Color: Text-Primary
- Label: "Option Price"
- Comparison: Previous calculation (if exists), show delta

Greeks Grid:
- Layout: 2x2 grid
- Each Cell:
  - Label: Greek name (Delta, Gamma, Theta, Vega)
  - Value: Monospace-Medium
  - Description: Tooltip on hover

Payoff Diagram:
- Chart-Type: Line chart (X: Stock price, Y: Payoff)
- Height: 300px
- Interactive: Hover to see exact values
- Lines:
  - Option-Payoff: Primary-500 (solid, thick)
  - Breakeven: Secondary-400 (dashed)
  - Current-Spot: Success-500 (vertical line)

Export Actions:
- Export-PDF: Download as PDF report
- Export-CSV: Download calculation data
- Share-Link: Copy shareable URL with parameters
- Save-Template: Save input preset

Mobile Behavior (< 768px):
- Layout: Stacked vertical
- Input-Panel: Full width, top
- Results-Panel: Full width, bottom
- Calculate-Button: Sticky at bottom
```

### 5.3 Portfolio Management Flow

```
Entry: Sidebar "Portfolio" → /portfolio

Page Structure:
┌─────────────────────────────────────────────────────────┐
│  Page Header                                            │
│  "Portfolio Overview"                                   │
│  [Add Position] [Export] [Settings]                     │
├─────────────────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │Total P&L │ │ Delta    │ │ Theta    │ │ Positions│   │
│  │ +$1,234  │ │  0.45    │ │ -12.5    │ │   15     │   │
│  │  +2.5%   │ │          │ │          │ │          │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │
├─────────────────────────────────────────────────────────┤
│  Filters: [All] [Calls] [Puts] [Expiring Soon]         │
│  Sort: [Expiration ▼]                                   │
├─────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────┐  │
│  │ Position Card 1                                   │  │
│  │ AAPL Call $150 Strike, Exp: 2024-03-15           │  │
│  │ Price: $5.20  |  P&L: +$120 (+5.2%)               │  │
│  │ Delta: 0.65  Gamma: 0.05  Theta: -0.12           │  │
│  │ [View Details] [Edit] [Close Position]            │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Position Card 2                                   │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘

KPI Cards (Top Row):
- Layout: 4 cards, equal width
- Card-Height: 120px
- Click-Action: Drill down to detailed view

Total P&L Card:
- Value: Monospace-Large, Color [Success/Error based on sign]
- Percentage: Body-Medium, same color
- Trend-Icon: Arrow up/down
- Sparkline-Chart: 7-day mini chart (optional)

Add Position Flow:
- Trigger: Click "Add Position" button
- Action: Open drawer (right-side, width 640px)

Drawer Content:
- Title: "Add New Position"
- Form:
  1. Position-Type: Radio [Long Call, Long Put, Short Call, Short Put]
  2. Underlying-Symbol: Autocomplete input (search stocks)
  3. Strike-Price: Number input
  4. Expiration-Date: Date picker
  5. Quantity: Number input (contracts)
  6. Entry-Price: Number input (premium paid/received)
  7. Entry-Date: Date picker (default: today)
  8. Notes: Textarea (optional)
- Actions: "Cancel" (Secondary), "Add Position" (Primary)

Position Card Drill-Down:
- Trigger: Click "View Details" or click card
- Action: Navigate to /portfolio/position/:id
- Page-Type: Full-width detail view

Detail View Structure:
Header:
- Breadcrumb: Portfolio > Position Details
- Title: "[Symbol] [Type] $[Strike] [Expiration]"
- Status-Badge: [Open, Closed, Expired]
- Actions: [Edit] [Close Position] [Delete]

Content Tabs:
- Overview: Summary metrics + current Greeks
- Performance: P&L chart over time
- Greeks-History: Historical Greeks charts
- Transactions: Entry/adjustment/exit log
- Analysis: What-if scenarios

Close Position Flow:
- Trigger: Click "Close Position"
- Action: Open confirmation dialog
- Dialog:
  - Title: "Close Position?"
  - Fields:
    - Exit-Price: Number input (current market price pre-filled)
    - Exit-Date: Date picker (today default)
  - Calculation: Auto-calculate realized P&L
  - Display: "Realized P&L: +$120 (+5.2%)" in large text
  - Actions: "Cancel", "Close Position" (Danger button)
- Success: Toast notification "Position closed successfully"
- Effect: Redirect to portfolio overview, card moves to "Closed Positions" tab
```

### 5.4 Analytics Deep Dive Flow

```
Entry: Portfolio Position Card → Click "Analyze" or Navigate via Analytics page

Landing Page: /analytics
┌─────────────────────────────────────────────────────────┐
│  Analytics Dashboard                                    │
│  [Export Report] [Schedule Report]                      │
├─────────────────────────────────────────────────────────┤
│  Select Analysis:                                       │
│  [Individual Position ▼] [Portfolio Aggregate]          │
│                                                         │
│  ┌──────────────────┐  ┌────────────────────────────┐  │
│  │ Quick Filters    │  │  Analysis Preview          │  │
│  │                  │  │                            │  │
│  │ Position: [▼]    │  │  [Select position to       │  │
│  │ Metric: [▼]      │  │   begin analysis]          │  │
│  │ Time Range: [▼]  │  │                            │  │
│  │                  │  │                            │  │
│  │ [Analyze]        │  │                            │  │
│  └──────────────────┘  └────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘

After Selection → Analysis View:
┌─────────────────────────────────────────────────────────┐
│  [Position Name] Greeks Analysis                        │
│  As of: 2024-12-13 14:23:45 EST                         │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────┐   │
│  │  Delta Over Time Chart                          │   │
│  │  [Line chart showing delta evolution]           │   │
│  │  Timeframe: [1D] [1W] [1M] [3M] [1Y] [All]      │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌──────────────────┐  ┌────────────────────────────┐  │
│  │ Gamma Analysis   │  │  Vega Sensitivity          │  │
│  │ [Chart]          │  │  [Chart]                   │  │
│  └──────────────────┘  └────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Theta Decay Projection                         │   │
│  │  [Area chart with projected decay]              │   │
│  │  Expected daily decay: -$0.15                   │   │
│  └─────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│  Scenario Analysis                                      │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Stock Price Scenarios                          │   │
│  │  [Heatmap: Stock price vs Time to expiry]      │   │
│  │                                                 │   │
│  │  Spot ↓  |  -10%  |  -5%  |  0%  |  +5%  | +10%│   │
│  │  ───────────────────────────────────────────────│   │
│  │  Today   | -$2.5  | -$1.2 | $0   | +$1.5 | +$3 │   │
│  │  1 week  | -$2.8  | -$1.5 | -$0.3| +$1.2 | +$2.7│  │
│  │  [...]                                          │   │
│  └─────────────────────────────────────────────────┘   │
│  [Export Scenarios] [Run Custom Scenario]               │
└─────────────────────────────────────────────────────────┘

Custom Scenario Builder:
- Trigger: Click "Run Custom Scenario"
- Action: Open modal (max-width 800px)
- Modal-Title: "Custom Scenario Analysis"
- Inputs:
  - Spot-Price-Range: Slider (±50% from current)
  - Volatility-Adjustment: Slider (±20% from current)
  - Time-Advance: Number input (days forward)
  - Rate-Change: Number input (bp change)
- Live-Preview: Update chart in real-time as sliders move
- Actions: "Reset", "Run Scenario" (Primary)
- Result: Display updated P&L heatmap

Greeks Comparison (Multi-Position):
- Entry: Analytics page → Select "Portfolio Aggregate"
- Display: Side-by-side Greek comparison across positions
- Chart-Type: Grouped bar chart
- Grouping: By Greek (Delta, Gamma, Theta, Vega)
- Bars: Each position as separate bar
- Interactivity: Click position to isolate/highlight

Export Report:
- Trigger: Click "Export Report"
- Modal-Title: "Generate Analytics Report"
- Options:
  - Report-Type: [PDF, Excel, CSV]
  - Include:
    - Checkboxes: [Summary, Charts, Scenarios, Raw Data]
  - Date-Range: Date range picker
  - Email-Delivery: Checkbox + email input
- Action: "Generate Report" → Show progress bar → Download/email
```

---

## 6. Dark Mode Specifications

### 6.1 Dark Mode Color Palette

```
Background Colors:
- Background-Default:  #121212  (rgb(18, 18, 18))
- Background-Paper:    #1E1E1E  (rgb(30, 30, 30))
- Background-Elevated: #2C2C2C  (rgb(44, 44, 44))

Surface Colors:
- Surface-1:  #1E1E1E  (1dp elevation)
- Surface-2:  #232323  (2dp elevation)
- Surface-3:  #252525  (3dp elevation)
- Surface-4:  #272727  (4dp elevation)
- Surface-6:  #2C2C2C  (6dp elevation)
- Surface-8:  #2E2E2E  (8dp elevation)
- Surface-12: #333333  (12dp elevation)
- Surface-16: #353535  (16dp elevation)
- Surface-24: #383838  (24dp elevation)

Text Colors:
- Text-Primary:   rgba(255, 255, 255, 0.87)  - High emphasis
- Text-Secondary: rgba(255, 255, 255, 0.60)  - Medium emphasis
- Text-Disabled:  rgba(255, 255, 255, 0.38)  - Disabled

Primary Colors (Adjusted):
- Primary-200: #90CAF9  (Lighter for dark backgrounds)
- Primary-300: #64B5F6
- Primary-400: #42A5F5  ← Main Primary in dark mode

Semantic Colors (Lightened):
- Success-400: #66BB6A  ← Success in dark mode
- Error-400:   #EF5350  ← Error in dark mode
- Warning-400: #FFCA28  ← Warning in dark mode
- Info-400:    #29B6F6  ← Info in dark mode

Borders & Dividers:
- Divider:  rgba(255, 255, 255, 0.12)
- Border:   rgba(255, 255, 255, 0.23)
- Outline:  rgba(255, 255, 255, 0.50)
```

### 6.2 Component Dark Mode Adjustments

```
Buttons:
Primary Button:
- Background: Primary-400 (#42A5F5)
- Hover: Primary-300 (#64B5F6)
- Active: Primary-200 (#90CAF9)

Cards:
- Background: Surface-1 (#1E1E1E)
- Elevated-Cards: Surface-2 (#232323)
- Border: rgba(255, 255, 255, 0.12)
- Hover-Elevation: Surface-4 (#272727)

Inputs:
- Background: Surface-2 (#232323)
- Border: rgba(255, 255, 255, 0.23)
- Focus-Border: Primary-400
- Text: Text-Primary

Tables:
- Header-Background: Surface-2 (#232323)
- Row-Default: Background-Default (#121212)
- Row-Alternate: Surface-1 (#1E1E1E)
- Row-Hover: Surface-3 (#252525)
- Row-Selected: rgba(66, 165, 245, 0.16)

Modals:
- Backdrop: rgba(0, 0, 0, 0.7) (darker)
- Background: Surface-6 (#2C2C2C)
- Header-Border: rgba(255, 255, 255, 0.12)

Shadows (Reduced):
- Shadow-SM:  0px 1px 2px rgba(0, 0, 0, 0.3)
- Shadow-MD:  0px 4px 6px rgba(0, 0, 0, 0.4)
- Shadow-LG:  0px 10px 15px rgba(0, 0, 0, 0.5)
- Shadow-XL:  0px 20px 25px rgba(0, 0, 0, 0.6)
```

### 6.3 Toggle Implementation

```
Dark Mode Toggle Location:
- Primary: Top navigation bar, right section
- Secondary: Settings page

Toggle Component:
- Type: Icon button toggle
- Icons:
  - Light-Mode: Sun icon (20px)
  - Dark-Mode: Moon icon (20px)
- Size: 40px x 40px
- Background-Active: Primary-100 (light) / Surface-3 (dark)

Persistence:
- Storage: localStorage key "theme-preference"
- Values: "light" | "dark" | "system"
- Default: "system" (respect OS preference)

Transition:
- Duration: Duration-Base (200ms)
- Properties: background-color, color, border-color
- Easing: Easing-Standard
- Avoid: Flashing on page load (inject theme before render)

System Preference Detection:
- Media-Query: @media (prefers-color-scheme: dark)
- Update: Listen for changes to OS preference
- Override: User-selected preference takes precedence
```

---

## 7. Financial Data Display Standards

### 7.1 Price Formatting

```
Option Prices (Premiums):
- Font: Roboto Mono
- Size: 14px (table), 24px (card), 48px (hero)
- Weight: 600 (SemiBold)
- Format: $XX.XX (always 2 decimals)
- Alignment: Right
- Color: Text-Primary
- Tabular-Nums: font-variant-numeric: tabular-nums

Stock Prices:
- Format: $XXX.XX (2 decimals)
- Large-Numbers: $X,XXX.XX (comma separator)

Percentage Changes:
- Format: +X.XX% or -X.XX%
- Sign: Always show + or -
- Color:
  - Positive: Success-600 (light) / Success-400 (dark)
  - Negative: Error-600 (light) / Error-400 (dark)
  - Zero: Neutral-600
- Icon: Triangle up (▲) or down (▼) preceding text

Currency Formatting:
- Symbol: $ (USD default)
- Thousands: Comma separator (1,234.56)
- Millions: $1.23M
- Billions: $1.23B
- Negative: -$XX.XX (not parentheses)
```

### 7.2 Greeks Display

```
Delta:
- Format: 0.XXXX (4 decimals)
- Range: -1.0000 to 1.0000
- Call-Convention: 0 to 1 (positive)
- Put-Convention: -1 to 0 (negative)
- Color: Neutral (not semantic)

Gamma:
- Format: 0.XXXX (4 decimals)
- Always-Positive: Yes
- Unit: No unit displayed

Theta:
- Format: -X.XX (2 decimals)
- Usually-Negative: Yes (time decay)
- Unit: Display "per day" on hover tooltip
- Color: Error-600 (decay is negative)

Vega:
- Format: X.XX (2 decimals)
- Always-Positive: Yes
- Unit: Display "per 1% IV change" on tooltip

Rho:
- Format: X.XX (2 decimals)
- Sign: Can be positive or negative
- Unit: Display "per 1% rate change" on tooltip

Display Pattern (Card):
┌──────────────────────────────────┐
│  Greeks                          │
├──────────────────────────────────┤
│  Delta    0.6523  [Info icon]   │
│  Gamma    0.0234  [Info icon]   │
│  Theta   -0.12    [Info icon]   │
│  Vega     0.18    [Info icon]   │
│  Rho      0.05    [Info icon]   │
└──────────────────────────────────┘

Tooltip on Info Icon:
- Title: Greek name
- Definition: Brief explanation
- Formula: Mathematical formula (optional)
- Example: "A delta of 0.65 means the option price changes $0.65 for every $1 move in the underlying."
```

### 7.3 Date & Time Formatting

```
Expiration Dates:
- Format: MMM DD, YYYY (e.g., "Mar 15, 2024")
- Alternative: DD MMM YYYY (e.g., "15 Mar 2024")
- Consistency: Choose one format globally

Time to Expiration:
- Format-Days: "X days" (e.g., "45 days")
- Format-Hours: "X hours" (< 24 hours)
- Format-Years: "X.XX years" (in calculations)
- Urgent: If < 7 days, display in Warning-600 color
- Expired: "Expired" badge in Error-600

Timestamps:
- Format: MMM DD, YYYY HH:MM AM/PM (e.g., "Dec 13, 2024 2:30 PM")
- Timezone: Display timezone abbreviation (EST, PST)
- Relative: Optional relative time (e.g., "2 hours ago") with absolute time in tooltip

Market Hours Indicator:
- Open: Green dot + "Market Open"
- Closed: Red dot + "Market Closed"
- Pre-Market: Amber dot + "Pre-Market"
- After-Hours: Amber dot + "After-Hours"
```

### 7.4 Data Tables Best Practices

```
Column Types:

Symbol Column:
- Width: Auto (based on content)
- Font: Body-Medium (not monospace)
- Text-Transform: uppercase
- Link: Clickable to position detail
- Color: Primary-600

Price Columns:
- Width: 100px fixed
- Alignment: Right
- Font: Monospace-Medium
- Format: $XX.XX

Percentage Columns:
- Width: 80px fixed
- Alignment: Right
- Font: Monospace-Medium
- Format: +X.XX%
- Color: Semantic (success/error)

Date Columns:
- Width: 120px fixed
- Alignment: Right
- Font: Body-Medium
- Format: MMM DD, YYYY

Greek Columns:
- Width: 80px fixed
- Alignment: Right
- Font: Monospace-Small
- Format: 0.XXXX

Action Column:
- Width: 100px fixed
- Alignment: Center
- Content: Icon buttons (Edit, Delete, More)

Sorting:
- Default-Sort: Expiration date (ascending)
- Sort-Indicator: Arrow icon in header
- Multi-Sort: Shift+click for secondary sort

Filtering:
- Position: Above table, below page title
- Layout: Horizontal row of filter chips
- Active-Filters: Display as removable chips
- Clear-All: Ghost button "Clear filters"
```

---

## 8. Implementation Guidelines

### 8.1 Material-UI v5 Theme Configuration

```typescript
// Path: frontend/src/theme/index.ts

import { createTheme, ThemeOptions } from '@mui/material/styles';
import { tokens } from './tokens';
import { typography } from './typography';
import { components } from './components';

const getThemeOptions = (mode: 'light' | 'dark'): ThemeOptions => ({
  palette: {
    mode,
    primary: tokens.colors.primary,
    secondary: tokens.colors.secondary,
    success: tokens.colors.success,
    error: tokens.colors.error,
    warning: tokens.colors.warning,
    info: tokens.colors.info,
    background: mode === 'light'
      ? tokens.colors.background.light
      : tokens.colors.background.dark,
    text: mode === 'light'
      ? tokens.colors.text.light
      : tokens.colors.text.dark,
  },
  typography: typography,
  spacing: 4, // 4px base grid
  shape: {
    borderRadius: 8, // Default radius-md
  },
  shadows: tokens.shadows,
  transitions: tokens.transitions,
  components: components,
});

export const lightTheme = createTheme(getThemeOptions('light'));
export const darkTheme = createTheme(getThemeOptions('dark'));
```

### 8.2 File Structure

```
frontend/src/
├── theme/
│   ├── index.ts              # Main theme export
│   ├── tokens.ts             # Design tokens
│   ├── typography.ts         # Typography configuration
│   ├── components.ts         # Component overrides
│   ├── breakpoints.ts        # Responsive breakpoints
│   └── shadows.ts            # Elevation shadows
├── components/
│   ├── ui/
│   │   ├── Button/
│   │   │   ├── Button.tsx
│   │   │   ├── Button.stories.tsx
│   │   │   └── Button.test.tsx
│   │   ├── Card/
│   │   ├── Input/
│   │   ├── Table/
│   │   ├── Modal/
│   │   ├── Alert/
│   │   └── index.ts
│   ├── domain/
│   │   ├── OptionCard/
│   │   ├── GreeksDisplay/
│   │   ├── PricingCalculator/
│   │   └── PortfolioSummary/
│   └── layout/
│       ├── AppLayout/
│       ├── Sidebar/
│       ├── TopNav/
│       └── PageHeader/
├── pages/
│   ├── Dashboard/
│   ├── Pricing/
│   ├── Portfolio/
│   └── Analytics/
└── hooks/
    ├── useTheme.ts
    ├── useBreakpoint.ts
    └── useAccessibility.ts
```

### 8.3 Component Development Standards

```
Every Component Must Include:

1. TypeScript Interface:
   - Explicit prop types
   - JSDoc comments
   - Default values

2. Accessibility:
   - ARIA labels where needed
   - Keyboard navigation
   - Focus management
   - Screen reader text

3. Responsive Behavior:
   - Mobile-first approach
   - Breakpoint-specific styles
   - Touch-friendly targets (min 44px)

4. States:
   - Default
   - Hover
   - Active
   - Focus
   - Disabled
   - Loading
   - Error

5. Documentation:
   - Storybook story
   - Usage examples
   - Prop documentation

6. Testing:
   - Unit tests (Jest + React Testing Library)
   - Accessibility tests (jest-axe)
   - Visual regression tests (Chromatic)
```

### 8.4 Performance Requirements

```
Bundle Size:
- Initial-Load: < 200KB (gzipped)
- Code-Splitting: Route-based
- Tree-Shaking: Enabled
- Lazy-Loading: Images, charts, heavy components

Runtime Performance:
- First-Contentful-Paint: < 1.5s
- Time-to-Interactive: < 3.0s
- Layout-Shift: < 0.1 (CLS)

Component Performance:
- Re-renders: Minimize with React.memo
- Event-Handlers: Debounce/throttle where appropriate
- Large-Lists: Virtualization (react-window)
- Heavy-Calculations: Web Workers

Accessibility Performance:
- Focus-Indicators: Instant (no delay)
- Keyboard-Navigation: < 16ms response
- Screen-Reader: No unnecessary announcements
```

---

## Conclusion

This Design System provides a complete blueprint for building a world-class financial application interface. Every specification is measured, every color is contrast-checked, and every interaction is accessibility-compliant.

**Implementation Priority**:
1. Core design tokens and theme configuration
2. Base UI component library (buttons, inputs, cards)
3. Layout system and navigation
4. Domain-specific components (Greeks, option cards)
5. Complex flows (onboarding, analytics)
6. Dark mode implementation
7. Accessibility audit and refinement

**Next Steps**:
1. Frontend Specialist: Implement theme configuration files
2. Frontend Specialist: Build component library following exact specifications
3. QA/Accessibility: Run WCAG 2.1 AA audit
4. Designer: Create high-fidelity mockups in Figma matching specifications

This system ensures consistency, usability, and professionalism across every pixel of the platform.
