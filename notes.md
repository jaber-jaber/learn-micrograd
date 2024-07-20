```python
class Value:
	def __init__(self, data, _children=(), _op='', label=''):
		self.data = data
		self._prev = set(_children) # prev is a set of all the children, children are tuples
        self._backward = lambda: None
        self._op = _op
        self.label = label

	def __repr__(self):
		return f"Value(data={self.data})"

	# add and mul are dunder methods which define how operations are handled for the Value object
	
	def __add__(self, other):
		out = Value(self.data + other.data, (self, other), 'add')
        def _backward(): 
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
		return out

	def __mul__(self, other):
		out = Value(self.data * other.data, (self, other), 'multiply')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward            
		return out

    def __rmul__(self,  other):
        return self * other # Python handles a * b differently to b * a (for the case that b is not a "Value" object but rather just an int/float)
        # here we're using rmul to right multiply the values (instead of make it commutative, just swap them)
        
    # division handling
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float pows for now
        # assert that we're handling an exponent that is integer/float
        out = Value(self.data ** other, (self, ), f'**{other}')

        def _backward():
            self.grad += other*(self**(other-1)) * out.grad
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        return self * other**-1
    
    # subtraction handling
    def __neg__(self):
        return self * -1
        
    def __sub__(self, other):
        return self + (-other)
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
        
        out._backward = _backward
        return out


    # = _backward = this function handles how derivates are computed for local operation that happens to the value

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        # Tanh is a hyperbolic tangent function that normalizes the output between 0 and 1.

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited: # if we haven't visited the node yet
                visited.add(v) # visited
                for child in v_prev:
                    build_topo(child) # do the same thing recursively for all children of the visited node

                topo.append(v) # add it to our topological cluster
        build_topo(self) 
        
        self.grad = 1.0

        for node in reversed(topo):
            node._backward()
        
        # This algorithm is called a Topological Sort Algorithm (we will go through each node which connects to the output and compute 
        # the gradient with respect to the output)
```
```python
def main():
	a = Value(3)
	b = Value(10)
	c = Value(2)

	d = a * b + c

if __name__ == "__main__":
	main()

```

Value object is scalar value => inputs into micrograd engine
micrograd is a backpropagation engine. Given a mathematical expression and its outputs, can we compute the effect of changes in the inputs to the output.

Example of a mathematical expression:
```python
a = 1.0
b = -4.0
e = 2.0

c = a * b
d = c + 2a
f = e * d
```

How can you backpropagate through this series of expressions to calculate the rate of change of 'f' by 'a', 'b' and 'e'.
1. We go step by step, iterating from f all the way to the inputs
2. Compute:
	1. Rate of change of d with respect to f
	2. c with respect to f
	3. a, b and with respect to f (chain rule)

This process is what makes neural networks work. By looking at the previous gradient, the network optimizes a loss function so that the gradients move in a positive direction. This will happen if the loss function approaches zero (no difference between last gradient step and current gradient step).

### How do operations work?
Locally, we need to think about what dc/da and dc/db are. dc/da = 1 * b and dc/db = a * 1.

Chain rule definition:
For the relationship between variables x and z, when z depends on y:
dz/dx = dy/dx * dz/dy

Hyperbolic tangent function definition:
tanhx = (e^2x - 1)/(e^2x+1)

## What am I going to do with this information?
- [ ] Implement micrograd but make it a bit more challenging? Maybe implement it in C, or build your own tensor library -- something to make it more difficult
