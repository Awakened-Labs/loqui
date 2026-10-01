#!/usr/bin/env python3
"""Generate modern, assistant-style sentences for tagger training.

The public-domain novels the tagger learns from have almost no currency
signs, clock times, version numbers, URLs or product names, which is most of
what an assistant reads aloud. These templates fill that gap. Every sentence
is generated here, so the text carries no third-party rights, and spaCy
supplies the tags (spacy_tag.py). corpus/assistant.txt is the held-out
evaluation set and is not reproduced by any template.

    synthetic.py N SEED > synthetic.txt
"""

import random
import sys

NAMES = ["Alice", "Priya", "Marcus", "Chen", "Olivia", "Diego", "Fatima", "Noah", "Hannah", "Kenji",
         "Sofia", "Omar", "Grace", "Liam", "Aisha", "Mateo", "Zoe", "Ethan", "Mia", "Lucas"]
PLACES = ["Denver", "Boston", "Seattle", "Austin", "Chicago", "Toronto", "London", "Berlin", "Tokyo",
          "Portland", "Atlanta", "Phoenix", "Dublin", "Madrid", "Oslo", "Lisbon"]
ORGS = ["Acme", "Globex", "Initech", "Umbrella", "Hooli", "Stark Industries", "Wayne Enterprises",
        "the city council", "the finance team", "the design team", "our landlord"]
TECH = ["the API", "the database", "the build", "the server", "the cache", "the router", "the app",
        "the website", "the deploy", "the pipeline", "the cluster", "the backup", "the laptop"]
FILES = ["config.toml", "README.md", "main.rs", "index.html", "notes.txt", "report.pdf", "data.csv",
         "settings.json", "Cargo.lock", "package.json"]
UNITS = ["kg", "km", "GB", "MB", "ms", "mph", "cm", "lbs", "miles", "hours", "minutes", "seconds"]
CURRENCY = ["$", "€", "£"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
VERBS_PAST = ["moved", "updated", "cancelled", "confirmed", "rescheduled", "approved", "declined",
              "shipped", "delivered", "renewed", "closed", "opened", "merged", "deployed"]
ACRONYMS = ["NASA", "FBI", "CEO", "API", "URL", "HTTP", "GPU", "CPU", "USB", "PDF", "SQL", "AWS", "UN", "EU"]
HETERONYMS = [
    "I will {v} the {n} tomorrow.", "Please {v} the meeting for Friday.", "The {n} was left on the desk.",
]
HET_PAIRS = [("record", "record"), ("present", "present"), ("object", "object"), ("project", "project"),
             ("permit", "permit"), ("contract", "contract"), ("produce", "produce"), ("conduct", "conduct"),
             ("refuse", "refuse"), ("desert", "desert"), ("lead", "lead"), ("read", "note"),
             ("close", "door"), ("live", "show"), ("wind", "clock"), ("tear", "page"), ("bow", "gift")]


def money(r):
    c = r.choice(CURRENCY)
    whole = r.choice([r.randrange(1, 100), r.randrange(100, 10000), r.randrange(10000, 10**7)])
    body = f"{whole:,}" if r.random() < 0.7 else str(whole)
    if r.random() < 0.5:
        body += f".{r.randrange(0, 100):02d}"
    return c + body


def clock(r):
    h, m = r.randrange(1, 13), r.choice([0, 5, 10, 15, 20, 30, 45, 50])
    ampm = r.choice([" AM", " PM", " a.m.", " p.m.", ""])
    return f"{h}:{m:02d}{ampm}"


def date(r):
    d = r.randrange(1, 29)
    suffix = {1: "st", 2: "nd", 3: "rd", 21: "st", 22: "nd", 23: "rd"}.get(d, "th")
    form = r.randrange(4)
    if form == 0:
        return f"{r.choice(MONTHS)} {d}{suffix}"
    if form == 1:
        return f"{r.choice(MONTHS)} {d}, {r.randrange(1990, 2035)}"
    if form == 2:
        return f"{r.randrange(1, 13)}/{d}"
    return f"{r.choice(DAYS)}, {r.choice(MONTHS)} {d}"


TEMPLATES = [
    lambda r: f"Your total comes to {money(r)} including tax.",
    lambda r: f"{r.choice(NAMES)} paid {money(r)} for the {r.choice(['tickets', 'repair', 'subscription', 'hotel', 'flight'])}.",
    lambda r: f"The invoice for {money(r)} is due on {date(r)}.",
    lambda r: f"Rent went up from {money(r)} to {money(r)} a month.",
    lambda r: f"The meeting starts at {clock(r)} on {r.choice(DAYS)}.",
    lambda r: f"Remind me at {clock(r)} to call {r.choice(NAMES)}.",
    lambda r: f"Your {r.choice(['train', 'flight', 'bus'])} leaves at {clock(r)} from gate {r.choice('ABCDE')}{r.randrange(1, 40)}.",
    lambda r: f"{r.choice(ORGS)} {r.choice(VERBS_PAST)} the order on {date(r)}.",
    lambda r: f"It was {r.randrange(-20, 105)} degrees in {r.choice(PLACES)} yesterday.",
    lambda r: f"About {r.randrange(1, 100)}% of {r.choice(['users', 'tests', 'orders', 'votes'])} {r.choice(['passed', 'failed', 'arrived', 'changed'])}.",
    lambda r: f"Version {r.randrange(0, 10)}.{r.randrange(0, 20)}.{r.randrange(0, 50)} fixes issue #{r.randrange(1, 5000)}.",
    lambda r: f"See issue #{r.randrange(1, 900)} and pull request #{r.randrange(1, 900)} for details.",
    lambda r: f"{r.choice(TECH).capitalize()} took {r.randrange(1, 900)} {r.choice(['ms', 'seconds', 'minutes'])} to respond.",
    lambda r: f"The file is {r.randrange(1, 90)}.{r.randrange(1, 10)} {r.choice(UNITS)} and lives in {r.choice(FILES)}.",
    lambda r: f"Open {r.choice(FILES)} and change the {r.choice(['port', 'path', 'token', 'timeout'])} setting.",
    lambda r: f"Call {r.choice(NAMES)} at 555-{r.randrange(1000, 10000)} before {clock(r)}.",
    lambda r: f"The {r.choice(ACRONYMS)} and the {r.choice(ACRONYMS)} released a joint statement.",
    lambda r: f"Ask the {r.choice(ACRONYMS)} team whether {r.choice(TECH)} supports {r.choice(ACRONYMS)}.",
    lambda r: f"{r.choice(NAMES)} scored {r.randrange(1, 100)} points in the {r.randrange(1, 5)}{['st', 'nd', 'rd', 'th'][r.randrange(4)]} quarter.",
    lambda r: f"This is the {r.randrange(2, 30)}th time {r.choice(TECH)} has crashed this {r.choice(['week', 'month'])}.",
    lambda r: f"We sold {r.randrange(1000, 10**6):,} units in {r.randrange(1995, 2030)}.",
    lambda r: f"The package weighs {r.randrange(1, 50)}.{r.randrange(1, 10)} {r.choice(['kg', 'lbs'])}.",
    lambda r: f"Set a timer for {r.randrange(1, 90)} minutes.",
    lambda r: f"What's the weather like in {r.choice(PLACES)} this {r.choice(['weekend', 'evening', 'morning'])}?",
    lambda r: f"Don't forget {r.choice(NAMES)}'s birthday on {date(r)}.",
    lambda r: f"I've {r.choice(VERBS_PAST)} your {r.choice(['booking', 'request', 'appointment', 'order'])}, and it's confirmed.",
    lambda r: f"Visit {r.choice(['example.com', 'docs.rs', 'github.com/org/repo', 'www.city.gov'])} for more.",
    lambda r: f"Send it to {r.choice(NAMES).lower()}@example.com by {r.choice(DAYS)}.",
    lambda r: f"The {r.choice(ORGS)} budget grew {r.randrange(1, 40)}% to {money(r)}.",
    lambda r: f"Q{r.randrange(1, 5)} revenue was {money(r)} million.",
    lambda r: f"{r.choice(NAMES)} and {r.choice(NAMES)} will meet at {r.randrange(1, 500)} {r.choice(['Main', 'Oak', 'Elm', 'Pine'])} Street.",
    lambda r: f"Mr. {r.choice(NAMES)} and Dr. {r.choice(NAMES)} arrive on the {r.randrange(1, 28)}th.",
    lambda r: (lambda v, n: r.choice(HETERONYMS).format(v=v, n=n))(*r.choice(HET_PAIRS)),
    lambda r: f"They {r.choice(['want', 'need', 'plan'])} to {r.choice(HET_PAIRS)[0]} it before {r.choice(DAYS)}.",
    lambda r: f"The {r.choice(HET_PAIRS)[1]} {r.choice(['was', 'is', 'looks'])} {r.choice(['ready', 'late', 'broken', 'fine'])}.",
    lambda r: f"Wi-Fi at the {r.choice(['café', 'airport', 'hotel', 'library'])} was {r.choice(['slow', 'down', 'fine'])}, so I used my phone.",
    lambda r: f"Here's a summary: {r.randrange(1, 9)} items are done and {r.randrange(1, 9)} are blocked.",
    lambda r: f"Run cargo test before you push; CI rejects failing builds.",
    lambda r: f"{r.choice(NAMES)} said, \"We'll ship it on {r.choice(DAYS)}.\"",
    lambda r: f"The score was {r.randrange(0, 10)}-{r.randrange(0, 10)} at halftime.",
]


def main():
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    r = random.Random(seed)
    seen = set()
    while len(seen) < n:
        s = r.choice(TEMPLATES)(r)
        if s not in seen:
            seen.add(s)
            print(s)


if __name__ == "__main__":
    main()
