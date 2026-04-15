package layer

import (
	"math"
	randv2 "math/rand/v2"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

// OptionFunc configures a LinearT layer.
type OptionFunc func(*LinearT)

// WithSource sets the random source used to initialize the weights.
func WithSource(s randv2.Source) OptionFunc {
	return func(l *LinearT) {
		l.s = s
	}
}

// WithInSize initializes the weight matrix with the given input size.
func WithInSize(inSize int) OptionFunc {
	return func(l *LinearT) {
		l.Add("w", initw(inSize, l.outSize, l.s))
	}
}

// WithNoBias removes the bias parameter from the layer.
func WithNoBias() OptionFunc {
	return func(l *LinearT) {
		l.Delete("b")
	}
}

// Linear returns a new linear layer with the given output size.
func Linear(outSize int, opts ...OptionFunc) *LinearT {
	p := make(Parameters)
	p.Add("b", variable.Zeros(1, outSize))

	l := &LinearT{
		outSize:    outSize,
		Parameters: p,
	}

	for _, opt := range opts {
		opt(l)
	}

	return l
}

// LinearT is a trainable linear layer.
type LinearT struct {
	outSize int
	s       randv2.Source
	Parameters
}

// First applies the layer and returns the first output.
func (l *LinearT) First(x ...*variable.Variable) *variable.Variable {
	return l.Forward(x...)[0]
}

// Forward applies the layer to x.
func (l *LinearT) Forward(x ...*variable.Variable) []*variable.Variable {
	if _, ok := l.Parameters["w"]; !ok {
		inSize := last(x[0].Shape())
		l.Add("w", initw(inSize, l.outSize, l.s))
	}

	return []*variable.Variable{
		F.Linear(l.xparams(x[0])...),
	}
}

func (l *LinearT) xparams(x *variable.Variable) []*variable.Variable {
	xp := []*variable.Variable{x, l.Parameters["w"]}
	if b, ok := l.Parameters["b"]; ok {
		xp = append(xp, b)
	}

	return xp
}

func initw(inSize, outSize int, s randv2.Source) *variable.Variable {
	w := tensor.Randn([]int{inSize, outSize}, s)
	xavier := 1.0 / math.Sqrt(float64(inSize))
	return variable.From(tensor.MulC(xavier, w))
}

func last(shape []int) int {
	return shape[len(shape)-1]
}
