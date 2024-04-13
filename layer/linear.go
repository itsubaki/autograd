package layer

import (
	"math"
	randv2 "math/rand/v2"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/rand"
	"github.com/itsubaki/autograd/variable"
)

type LinearOpts struct {
	InSize int
	NoBias bool
	Source randv2.Source
}

func Linear(outSize int, opts ...LinearOpts) *LinearT {
	s := rand.NewSource(rand.MustRead())
	if len(opts) != 0 && opts[0].Source != nil {
		s = opts[0].Source
	}

	p := make(Parameters)
	if len(opts) == 0 || !opts[0].NoBias {
		p.Add("b", variable.Zero(1, outSize))
	}
	if len(opts) != 0 && opts[0].InSize > 0 {
		p.Add("w", initw(opts[0].InSize, outSize, s))
	}

	return &LinearT{
		outSize:    outSize,
		source:     s,
		Parameters: p,
	}
}

type LinearT struct {
	outSize int
	source  randv2.Source
	Parameters
}

func (l *LinearT) First(x ...*variable.Variable) *variable.Variable {
	return l.Forward(x...)[0]
}

func (l *LinearT) Forward(x ...*variable.Variable) []*variable.Variable {
	if _, ok := l.Parameters["w"]; !ok {
		inSize := variable.Shape(x[0])[1]
		l.Parameters.Add("w", initw(inSize, l.outSize, l.source))
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
