# LocalGPT UI fonts

These font files are vendored locally so the LocalGPT UI works fully offline
(no CDN dependency after installation).

## Families

| Family      | Weights / styles                      | Used for              |
|-------------|---------------------------------------|-----------------------|
| Lato        | 300, 400, 700, 900 + 400/700 italic   | "Lato" font option    |
| EB Garamond | variable 400–800 (roman + italic)     | "EB Garamond" option  |

Only the `latin` and `latin-ext` subsets are vendored.

## Source and license

Downloaded from [Google Fonts](https://fonts.google.com) (Lato by Łukasz Dziedzic,
EB Garamond by Georg Duffner, Octavio Pardo). Both families are licensed under the
SIL Open Font License 1.1 — see <https://openfontlicense.org>.

## Regenerating

```
python scripts/download_fonts.py
```

This re-downloads the woff2 files and rewrites `app/static/css/fonts.css`.
