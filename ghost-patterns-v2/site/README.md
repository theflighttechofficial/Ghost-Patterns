# Ghost Patterns website

A static site with no build step: plain HTML, one stylesheet and two scripts.

| Page | Contents |
|---|---|
| `index.html` | Landing page with a live chart computed in the browser |
| `lab.html` | Interactive experiment: plant a rule in noise and validate it |
| `method.html` | The method, formulas and null models |
| `results.html` | Experiments, case studies and limits |
| `docs.html` | Install, command line, options, outputs, Python API |
| `404.html` | Not-found page |

Shared files live in `assets/`: `css/style.css`, `js/site.js` (theme toggle,
menu, copy buttons) and `js/ghost.js` (the in-browser experiment and charts).

## Preview locally

```bash
python -m http.server 8000 --directory site
```

Then open http://localhost:8000.

## Publish

The workflow `.github/workflows/pages.yml` at the repository root deploys this
folder to GitHub Pages on every push that changes it. Enable it once under
**Settings → Pages → Source: GitHub Actions**. Any static host (Netlify,
Vercel, Cloudflare Pages) also works: point it at this folder, with no build
command.
