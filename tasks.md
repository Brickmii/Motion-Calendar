# Current Task: Fix Existence Dot (Dot 2)

## Context
This is a Blender simulation of the Motion Calendar framework — a model of how existence emerges from initial heat through polarity.

## The sequence (how it should work)
1. **Heat dot** — A spinning dot appears. This is the original heat.
2. **Polarity division** — The heat divides itself into positive and negative, creating a polarity line.
3. **Existence dot** — A NEW spinning dot appears at the exact midpoint of the polarity line. This dot represents existence itself. It is the same substance as the line, but because polarity now exists, this dot can be differentiated from the line. It is neither positive nor negative — it is neutral, sitting at the center of both.

## The bug
Dot 2 (the existence dot) currently moves outward to the position of cubic expansion. This is wrong.

## The fix
- Dot 2 should appear at the **midpoint of the polarity line** (between positive and negative)
- It should be **spinning** (same as the heat dot)
- It should **stay at the center** — it does NOT move outward to cubic expansion
- It is neutral — visually it should be distinguishable from both the positive and negative ends of the polarity line

## What NOT to change
- Dot 1 (the original heat dot) — leave as is
- The polarity line and its division behavior — leave as is
- Everything else in the scene — leave as is

## How to test
- Run the simulation in Blender
- Verify the existence dot appears at the polarity midpoint
- Verify it spins but does not translate/move outward
- Verify the rest of the sequence is unaffected
