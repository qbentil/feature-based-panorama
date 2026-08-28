# Sample images

These folders hold **real University of Ghana campus photographs** from
[Wikimedia Commons](https://commons.wikimedia.org/). Authors and licences are
in [`ATTRIBUTION.md`](ATTRIBUTION.md).

They replace the earlier synthetic murals so the app shows Legon buildings.
They were **not** captured as a dedicated left-to-right panorama (30–50%
overlap from one spot). Matching will be harder than on the synthetic set;
that is expected.

Official pages such as [ug.edu.gh](https://ug.edu.gh/) and
[dcs.ug.edu.gh](https://dcs.ug.edu.gh/) are all-rights-reserved. This project
does not copy those files. `dec.ug.edu.gh` does not currently resolve.

## What is in each folder

| Folder | Contents | Why it is here |
|---|---|---|
| `scenes/scene_main/` | 3 Balme Library views | Primary panorama (strong overlap on the facade) |
| `scenes/scene_viewpoint/` | Dance Department: street, courtyard, street | Viewpoint change |
| `scenes/scene_lighting/` | Three Night Market stall views | Related outdoor pair for lighting table |
| `scenes/scene_second/` | 5 Commonwealth Hall walk-up views | Harder extra scene |

Rotation and scale robustness are **not** extra folders. Those are applied in
code from a main pair (15/30/45/90° and 0.5× / 0.75× / 1.5×).

## How to capture your own photos

- Phone camera is fine. Same zoom for a given scene. Hold the phone level.
- Stand in one place and rotate your body. Do not walk large steps sideways
  except for the viewpoint folder.
- Aim for **30–50% overlap** between neighbouring shots.
- Name files `01.jpg`, `02.jpg`, … left to right.

Replace images in the same folder names. The app and experiment script pick up
whatever JPEGs or PNGs are in the folder.

## Re-download the Commons set

```bash
python scripts/download_ug_commons.py
```
