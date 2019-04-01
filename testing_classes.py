class test:
    """docstring for test."""
    a = 5
    b = 10
    def __init__(self, a,b):
        # super(, self).__init__()
        self.a = a
        self.b = b

print test.a
print test.b

X = test(10,20)
print X.a
print X.b
