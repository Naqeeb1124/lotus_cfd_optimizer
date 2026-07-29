import sys
import ansys.fluent.core as pyfluent

print(sys.executable)
print(pyfluent.__file__)
print(pyfluent.__path__)
print(hasattr(pyfluent, "launch_fluent"))