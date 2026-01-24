import mediapipe as mp
print('module file:', getattr(mp, '__file__', None))
print('version:', getattr(mp, '__version__', 'unknown'))
print('has_solutions:', hasattr(mp, 'solutions'))
print('dir snippet:', [k for k in dir(mp) if k.startswith('sol') or k.startswith('tasks') or k.startswith('Image')])
