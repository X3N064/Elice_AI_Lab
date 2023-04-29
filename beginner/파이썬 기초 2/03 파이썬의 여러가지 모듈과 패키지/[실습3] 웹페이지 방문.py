## 이렇게 해보세요!를 따라 수행해보세요.
from urllib.request import urlopen

webpage= urlopen("https://en.wikipedia.org/wiki/Lorem_ipsum").read().decode("utf-8")
print(webpage)