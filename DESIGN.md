# Yuhan Deng Academic Blog Design System

## 1. Atmosphere & Identity

The site feels like a carefully annotated research notebook: calm, exact, and
approachable to readers who are meeting a technical idea for the first time.
Its signature is the **visual proof**: article figures do not decorate a concept
or merely list its modules; each figure stages the phenomenon, mechanism, and
conclusion so the reader can reconstruct the argument from the image alone.

## 2. Color

### Palette

| Role              | Token                       | Light            | Dark      | Usage                                 |
| ----------------- | --------------------------- | ---------------- | --------- | ------------------------------------- |
| Surface/primary   | `--global-bg-color`         | `#ffffff`        | `#1c1c1d` | Page background                       |
| Surface/figure    | `--figure-bg`               | `#f4f8f7`        | `#101a1a` | SVG stage                             |
| Surface/elevated  | `--figure-panel`            | `#ffffff`        | `#182525` | Diagram objects                       |
| Text/primary      | `--global-text-color`       | `#000000`        | `#eeeeee` | Body and figure labels                |
| Text/secondary    | `--global-text-color-light` | `#828282`        | `#a7b0b0` | Captions and annotations              |
| Border/default    | `--global-divider-color`    | `rgba(0,0,0,.1)` | `#424246` | Dividers and figure outlines          |
| Accent/teal       | `--global-theme-color`      | `#0f766e`        | `#5eead4` | Main process and verified state       |
| Accent/teal-soft  | `--figure-teal-soft`        | `#ccfbf1`        | `#134e4a` | Teal fills                            |
| Accent/blue       | `--figure-blue`             | `#0369a1`        | `#7dd3fc` | Content, input, and client-side state |
| Accent/blue-soft  | `--figure-blue-soft`        | `#e0f2fe`        | `#0c4a6e` | Blue fills                            |
| Accent/amber      | `--figure-amber`            | `#b45309`        | `#fbbf24` | Position, gating, credentials         |
| Accent/amber-soft | `--figure-amber-soft`       | `#fef3c7`        | `#78350f` | Amber fills                           |
| Accent/coral      | `--figure-coral`            | `#c2410c`        | `#fb923c` | Conflict, mismatch, blocked paths     |
| Accent/coral-soft | `--figure-coral-soft`       | `#ffedd5`        | `#7c2d12` | Coral fills                           |

### Rules

- Teal means a valid connection, preserved relationship, or final conclusion.
- Blue means input content or a source-side object.
- Amber means coordinates, selection, credentials, or a transformation control.
- Coral is reserved for ambiguity, mismatch, or a path that will not work.
- Every figure supports light and dark color schemes with the same semantic roles.

## 3. Typography

### Scale

| Level        | Size                         | Weight | Line height | Tracking   | Usage                    |
| ------------ | ---------------------------- | ------ | ----------- | ---------- | ------------------------ |
| Display      | `clamp(2.2rem, 6vw, 3.6rem)` | 700    | 1.12        | `-0.045em` | Post title               |
| H2           | `1.55rem`                    | 600    | 1.35        | `-0.01em`  | Article sections         |
| H3           | `1.25rem`                    | 600    | 1.4         | `0`        | Subsections              |
| Body         | `1.02rem`                    | 400    | 1.9         | `0`        | Article prose            |
| Figure/title | `1.5rem` equivalent          | 700    | 1.25        | `-0.01em`  | Figure claim             |
| Figure/label | `0.95rem` equivalent         | 600    | 1.4         | `0`        | Object labels            |
| Figure/note  | `0.8rem` equivalent          | 400    | 1.45        | `0`        | Evidence and annotations |
| Caption      | `0.875rem`                   | 400    | 1.5         | `0`        | Figure captions          |

### Font Stack

- Primary: `-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", Arial, sans-serif`
- Mono: `"SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace`

### Rules

- Chinese figure labels use the site sans stack and stay at least 13px in the
  960px SVG viewBox.
- Equations, dimensions, file names, and commands use the mono stack when useful.
- The figure title states the conclusion, not the topic.

## 4. Spacing & Layout

### Base Unit

All authored spacing derives from 4px. Existing theme spacing remains the source
of truth for page layout.

| Token        | Value | Usage                   |
| ------------ | ----- | ----------------------- |
| `--space-1`  | 4px   | Fine diagram gaps       |
| `--space-2`  | 8px   | Label-to-object gap     |
| `--space-3`  | 12px  | Compact figure padding  |
| `--space-4`  | 16px  | Standard gap            |
| `--space-6`  | 24px  | Figure group gap        |
| `--space-8`  | 32px  | Section separation      |
| `--space-12` | 48px  | Major visual separation |

### Grid

- Site max width: 930px; article reading column: 760px.
- Figures use a 960px-wide SVG viewBox and a content-safe inset of at least 32px.
- Reading order is left-to-right or top-to-bottom, never both at once without
  numbered annotations.
- At narrow widths, dense explainer figures keep their 960px canvas inside a
  focusable horizontal viewport instead of shrinking labels below legibility.
  Touch, trackpad, mouse wheel, and keyboard users can pan the same figure; the
  surrounding caption remains in the article column.

## 5. Components

### Article figure

- **Structure**: semantic `figure` containing `picture`, responsive `img`, and
  `figcaption`.
- **Variants**: eager first figure; lazy subsequent figures.
- **Spacing**: caption uses `--space-3` above and `--space-6` below.
- **States**: light and dark color schemes; reduced-motion mode; mobile
  horizontal-pan viewport for dense explainers.
- **Accessibility**: meaningful `alt`, internal SVG `title` and `desc`, no
  information represented by color alone; the mobile pan viewport is a labeled,
  keyboard-focusable region.
- **Motion**: optional and finite; only reveals flow, correspondence, or a
  transformation already visible in the static frame.
- **Layout**: stack; document owns scrolling.

### Explainer SVG

- **Structure**: claim title, one dominant scene, short annotations, and a final
  takeaway band or highlighted proof.
- **Variants**: process, comparison, geometry, and relationship map.
- **Spacing**: 32px safe inset, 16px object gaps, 24px group gaps.
- **States**: light, dark, and reduced motion.
- **Accessibility**: `role="img"`, unique `title`/`desc`, 3:1 graphical-object
  contrast, 4.5:1 label contrast, patterns or labels supplement color.
- **Motion**: transform and opacity only; no infinite loops.
- **Layout**: a single visual path with numbered steps when more than two stages
  are present.

### Diagram vocabulary

- **Structure**: scene/object, semantic connector, evidence annotation, result.
- **Variants**: solid connector for data flow, dashed connector for challenge or
  hypothetical mapping, bracket for invariant dimensions, coral cross for a
  blocked or mismatched operation.
- **States**: static by default; optional finite reveal.
- **Accessibility**: every connector has a nearby text label; color is redundant.
- **Motion**: moving markers only when they clarify direction.
- **Layout**: connectors never cross labels and never imply an unlabeled branch.

## 6. Motion & Interaction

| Type          | Duration   | Easing                        | Usage                             |
| ------------- | ---------- | ----------------------------- | --------------------------------- |
| Micro         | 140ms      | ease-out                      | Existing controls                 |
| Standard      | 240ms      | ease-in-out                   | Existing navigation or card state |
| Figure reveal | 900-1800ms | cubic-bezier(0.16, 1, 0.3, 1) | One finite explanatory transition |

- SVG motion is never required to understand the figure.
- Only `transform`, `opacity`, and stroke-dash reveal are used.
- `prefers-reduced-motion: reduce` presents the final static state.
- Non-interactive figure objects do not animate merely for decoration.

## 7. Depth & Surface

The strategy is **mixed, restrained editorial depth**. Page components use the
existing borders and soft shadows. Figures use a tinted paper-like stage, thin
outlines, and one subtle lifted focal object; they avoid repeated floating cards.
Depth comes primarily from overlap, scale, and tonal grouping rather than heavy
drop shadows.

## 8. Accessibility Constraints & Accepted Debt

### Constraints

- WCAG target: 2.2 AA, with 4.5:1 body/label contrast and 3:1 meaningful graphic
  contrast.
- Figures remain understandable without motion and without distinguishing hue.
- Alt text states the relationship or conclusion, not merely “diagram”.
- Chinese labels are concise enough to remain readable at 375px viewport width.
- Dark mode uses explicitly authored colors rather than filter-based inversion.

### Inclusive personas

- **First-time technical reader**: can identify the starting problem and final
  conclusion without knowing every symbol.
- **Reader scanning on a phone**: can follow the dominant visual path without
  zooming to read paragraph-length labels.
- **Reader using a screen reader**: receives the same argument through figure alt
  text plus the surrounding caption and prose.
- **Reader with reduced-motion preference**: sees the completed proof with no
  missing state.

### Accepted Debt

| Item | Location | Why accepted | Owner / Exit |
| ---- | -------- | ------------ | ------------ |
| None | —        | —            | —            |
