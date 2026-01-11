"""
Run this to watch the Motion Engine simulation.

Controls:
  SPACE  - Pause/resume
  S      - Spawn new entity at mouse
  R      - Reset everything
  V      - Toggle 3D volume view
  +/-    - Speed up/slow down
  Click  - Select entity to see its stats
  Q      - Quit
"""

from motion_engine_two_deaths import MotionEngine, PygameRenderer

# Create universe
engine = MotionEngine(800, 600)

# Spawn 40 entities
for _ in range(40):
    engine.spawn()

# Open the window and run
renderer = PygameRenderer(engine, width=1000, height=700)
renderer.run()
