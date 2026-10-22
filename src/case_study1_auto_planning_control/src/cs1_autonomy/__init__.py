"""CS1 autonomous path control; shares hardware transport with the manual package."""
from pathlib import Path
PACKAGE = Path(__file__).resolve().parents[2]
if not (PACKAGE/'package.xml').exists():
    import rospkg
    PACKAGE = Path(rospkg.RosPack().get_path('case_study1_auto_planning_control'))
