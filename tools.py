import platform

# Getting system information
system = platform.system()
node = platform.node()
release = platform.release()
version = platform.version()
machine = platform.machine()
processor = platform.processor()

# Equivalent to 'uname -a'
uname_info = f"{system} {node} {release} {version} {machine} {processor}"