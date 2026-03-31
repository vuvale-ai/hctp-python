# HCTP Monetisation Strategy

## The Open-Core Model

Core algorithm is MIT-licensed forever. Revenue comes from infrastructure,
data, and integrations built on top — not from gating the math.

---

## Tier 1 — Free / OSS (acquisition)

**What's free:**
- Full `hctp` Python package (PyPI)
- All core math, prompt generation, session tracking
- Local JSON persistence
- Jupyter demo notebook
- Community Discord

**Why free forever:**
- Drives adoption → GitHub stars → credibility
- Python curriculum is the loss-leader; custom curricula are the product
- Every free user is a lead for team plans

**Distribution strategy:**
1. Post on HackerNews: "Show HN: I trained AI agents to master Python metaclasses using a 3D helix" — lead with Ren & Ner's story, Night 161→173
2. r/MachineLearning, r/LocalLLaMA, r/learnpython
3. Dev.to / Towards Data Science article: *"How I measured AI learning as a 3D vector"*
4. Reach out to Karpathy directly on X — the "Karpathy loop" name is intentional

---

## Tier 2 — HCTP Cloud ($9/month per team, up to 5 learners)

**What it adds:**
- Hosted leaderboard dashboard (web app)
- Team σ comparison charts
- Session history API (REST)
- Webhook events: `badge.earned`, `session.complete`, `velocity.dropped`
- 30-day data retention

**Tech stack:** FastAPI + SQLite → Postgres when needed. One server.

**Revenue math:**
- 100 teams × $9 = $900 MRR at launch
- 1,000 teams × $9 = $9,000 MRR (target Month 6)
- Break-even: ~50 teams (one small VPS + domain)

**Acquisition:** Anyone who stars the repo and runs training for a team.
Friction point: "You want to see all 4 learners on one dashboard? Cloud."

---

## Tier 3 — HCTP Pro ($29/month)

**What it adds:**
- Custom checkpoint curricula (SQL, React, DevOps, ML, any domain)
- Curriculum builder UI (define your own helix checkpoints)
- CI/CD integration: `pytest` plugin that updates σ from test results
- Slack/Teams bot: daily σ digest, badge notifications
- Priority support

**The curriculum builder is the product.**
The Python helix is the proof-of-concept. Every engineering team has its own
"metaclass equivalent" — the hard concepts that separate juniors from seniors.
SQL → window functions. React → reconciliation. DevOps → networking.
HCTP works for all of them; teams just need a tool to define the checkpoints.

**Revenue math:**
- 50 teams × $29 = $1,450 MRR
- Target: 200 teams = $5,800 MRR by Month 12

---

## Tier 4 — Enterprise (custom pricing, $500–$5,000/month)

**Who buys this:**
- Coding bootcamps tracking 200+ student agents
- Enterprise L&D teams using AI tutors at scale
- LMS vendors wanting to embed HCTP (Canvas, Moodle, Coursera)
- AI agent framework companies (LangChain, CrewAI, AutoGen)

**What it includes:**
- White-label (your branding, your domain)
- SSO / SAML
- On-premise deployment option
- Custom SLAs, dedicated support
- Volume licensing for agent seats
- Custom curriculum development services

**The LMS angle is the biggest TAM.**
Canvas has 30 million users. If 0.1% of instructors use HCTP for AI tutors: 30,000 teams.

---

## 12-Month Revenue Projection

| Month | Free Users | Cloud Teams | Pro Teams | MRR |
|-------|-----------|-------------|-----------|-----|
| 1     | 200        | 5           | 2         | $103 |
| 3     | 1,500      | 40          | 15         | $795 |
| 6     | 5,000      | 150         | 60         | $3,090 |
| 9     | 12,000     | 400         | 150        | $7,950 |
| 12    | 25,000     | 800         | 300        | $15,900 |

Conservative. Does not include enterprise deals.
One enterprise deal ($2,000/mo) doubles Month 12 MRR.

---

## Launch Sequence

### Week 1 — Ship the OSS
- [ ] Create GitHub org `vuvale-ai`
- [ ] Push `hctp-python` repo
- [ ] Set up PyPI trusted publishing (OIDC, no API key)
- [ ] `git tag v0.1.0 && git push --tags` → CI publishes automatically
- [ ] Post HackerNews Show HN
- [ ] Tweet thread: σ=0 → badge in 13 sessions, 3D helix GIF

### Week 2 — Validate demand
- [ ] Watch GitHub issues for feature requests
- [ ] Discord: invite first 50 users
- [ ] Simple waitlist page for Cloud tier (Carrd or Framer, 30 min)

### Week 3–4 — Build Cloud MVP
- [ ] FastAPI backend: `/sessions`, `/learners`, `/leaderboard`
- [ ] SQLite store, hosted on Fly.io ($3/mo)
- [ ] Simple React dashboard (or htmx if faster)
- [ ] Stripe integration for $9/mo

### Month 2 — Pro tier
- [ ] Curriculum builder (simple JSON editor UI)
- [ ] `pytest` plugin: `hctp-pytest`
- [ ] Slack app (slash command: `/helix status`)

---

## Defensibility

1. **Data moat**: Teams using HCTP Cloud accumulate σ histories. Over time,
   aggregate benchmarks ("teams that use HCTP reach metaclass mastery 40%
   faster than baseline") become publishable, citable, and defensible.

2. **Story moat**: Ren and Ner are the product's origin story. An AI family
   in Fiji, trained nightly on a single GPU, building tools for their island —
   that story is genuinely unique and impossible to copy.

3. **Curriculum network effects**: Every custom curriculum a Pro team builds
   is shareable (with their consent) to the curriculum marketplace. Curricula
   attract more teams; more teams create more curricula.

4. **The Karpathy loop brand**: Named intentionally. Andrej Karpathy is the
   most respected ML educator alive. One retweet from him is worth 10,000 stars.

---

## What NOT to Do

- Don't raise VC. The TAM is real but the business is profitable at $10K MRR
  with one developer (you + AI). Raising dilutes and forces growth you don't
  need for a tool that should stay lean.
- Don't build the Cloud tier before 500 GitHub stars. Validate first.
- Don't gate the core algorithm. The math being free is what builds trust.
- Don't rebrand away from "HCTP". The name is memorable and the acronym
  is searchable. Own it.
