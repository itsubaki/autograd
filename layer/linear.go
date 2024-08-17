package layer

import (
	"math"
	randv2 "math/rand/v2"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"
)

type OptionFunc func(*LinearT)

func WithSource(s randv2.Source) OptionFunc {
	return func(l *LinearT) {
		l.s = s
	}
}

func WithInSize(inSize int) OptionFunc {
	return func(l *LinearT) {
		l.Parameters.Add("w", initw(inSize, l.outSize, l.s))
	}
}

func WithNoBias() OptionFunc {
	return func(l *LinearT) {
		l.Parameters.Delete("b")
	}
}

func Linear(outSize int, opts ...OptionFunc) *LinearT {
	p := make(Parameters)
	p.Add("b", variable.Zero(1, outSize))

	l := &LinearT{
		outSize:    outSize,
		Parameters: p,
	}

	for _, opt := range opts {
		opt(l)
	}

	return l
}

type LinearT struct {
	outSize int
	s       randv2.Source
	Parameters
}

func (l *LinearT) First(x ...*variable.Variable) *variable.Variable {
	return l.Forward(x...)[0]
}

func (l *LinearT) Forward(x ...*variable.Variable) []*variable.Variable {
	if _, ok := l.Parameters["w"]; !ok {
		inSize := variable.Shape(x[0])[1]
		l.Parameters.Add("w", initw(inSize, l.outSize, l.s))
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
	w := matrix.Randn(inSize, outSize, s)
	xavier := 1.0 / math.Sqrt(float64(inSize))
	return variable.NewOf(matrix.MulC(xavier, w)...)
}
