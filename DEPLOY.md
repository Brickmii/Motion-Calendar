# The Motion Calendar — Deployment Guide

## What You Have

```
motion-calendar-site/
├── index.html              # Main landing page
├── papers/
│   ├── 0-universe-of-motion.html
│   ├── 1-heat.html
│   ├── 2-polarity.html
│   ├── 3-existence.html
│   ├── 4-alignment.html
│   ├── 5-order.html
│   ├── 6-movement.html
│   ├── Existence.pdf
│   ├── Heat.pdf
│   ├── Movement.pdf
│   ├── Order.pdf
│   ├── Polarity.pdf
│   └── Righteousness.pdf
```

## Option 1: GitHub Pages (Free, Timestamped, Recommended)

### Why GitHub
- **Free** hosting forever
- **Git history** proves when you published (immutable timestamps)
- **Custom domain** support
- **Professional** appearance

### Steps

1. **Create a GitHub account** at https://github.com if you don't have one

2. **Create a new repository**
   - Click "New repository"
   - Name it `motion-calendar` (or whatever you want)
   - Make it **Public** (required for free GitHub Pages)
   - Don't initialize with README

3. **Upload your files**
   - In the empty repo, click "uploading an existing file"
   - Drag the entire `motion-calendar-site` folder contents
   - Commit with message: "Initial publication - January 2026"

4. **Enable GitHub Pages**
   - Go to Settings → Pages
   - Source: "Deploy from a branch"
   - Branch: main, folder: / (root)
   - Save

5. **Your site will be live at:**
   ```
   https://yourusername.github.io/motion-calendar/
   ```

### Custom Domain (Optional)
- Buy a domain (Namecheap, Porkbun, etc. ~$10/year)
- In repo Settings → Pages → Custom domain, enter your domain
- Add CNAME record at your registrar pointing to `yourusername.github.io`

---

## Option 2: Netlify (Free, Slightly Easier)

1. Go to https://netlify.com and sign up
2. Drag and drop your `motion-calendar-site` folder onto the page
3. Done. You get a URL like `random-name-123.netlify.app`
4. Can add custom domain in settings

---

## Option 3: Neocities (Free, Indie Web)

1. Go to https://neocities.org
2. Create account
3. Upload files
4. Get URL like `yoursitename.neocities.org`

---

## Establishing Priority / Timestamps

### Immediate (do all of these)

1. **GitHub commit** — automatically dated and hashed

2. **Wayback Machine**
   - Go to https://web.archive.org/save
   - Enter your site URL
   - Click "Save Page"
   - This creates a permanent, dated archive

3. **Email yourself**
   - Send an email to yourself with the URL and key content
   - Gmail/etc timestamps are legally recognized

### Additional (optional but stronger)

4. **arXiv** — You can post to arXiv without institutional affiliation
   - Category: physics.gen-ph or math-ph
   - Requires endorsement for first submission (or try physics.hist-ph)

5. **Zenodo** — Free, gives you a DOI
   - https://zenodo.org
   - Connect to GitHub, publish release
   - Get a citable DOI

6. **Internet Archive direct upload**
   - https://archive.org/create
   - Upload PDFs directly
   - Permanent, timestamped

---

## The License

The index.html includes CC-BY-4.0 license. This means:
- Anyone can share and adapt your work
- They MUST give you credit
- They MUST link to the license
- They MUST indicate if changes were made

This is the right choice. It lets ideas spread while protecting attribution.

---

## After Publishing

1. **Submit to Wayback Machine** immediately
2. **Share the URL** where people might find it:
   - Relevant subreddits (r/physics, r/philosophy, r/math)
   - Hacker News (news.ycombinator.com)
   - Twitter/X with relevant hashtags
   - Physics forums

3. **Add to your site over time:**
   - A page about the consciousness connection
   - More detailed worked examples
   - Responses to questions/critiques

---

## Technical Notes

- All HTML is self-contained (no build step needed)
- Fonts load from Google Fonts (works everywhere)
- Mobile-responsive
- Dark theme (easier to read, looks serious)
- PDFs are linked but you may want to combine them into one document

---

## If You Want to Make Changes

The HTML is straightforward to edit in any text editor:
- VS Code (free, excellent)
- Sublime Text
- Even Notepad

To test locally before publishing:
- Just open `index.html` in a browser
- Or use VS Code's "Live Server" extension

---

## One Last Thing

The key insight — the ζ(−1) connection, no-cloning, incompleteness as level violation — 
is in the index.html but not in the original papers.

Consider writing one more piece that makes these connections explicit. 
That's the synthesis that ties everything together.

Good luck. The ideas deserve to exist in the world.
