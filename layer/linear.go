package layer

import (
	"math"
	"math/rand"
	"time"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"
)

type LinearOpts struct {
}

func Linear(outSize int, s ...rand.Source) *Layer {
	if len(s) == 0 {
		s = append(s, rand.NewSource(time.Now().UnixNano()))
	}

	p := make(Parameters)
	p.Add("b", variable.Zero(1, outSize))

	return &Layer{
		Forwarder: &LinearT{
			outSize:    outSize,
			rnd:        rand.New(s[0]),
			Parameters: p,
		},
	}
}

type LinearT struct {
	inSize, outSize int
	rnd             *rand.Rand
	Parameters
}

func (l *LinearT) Forward(x ...*variable.Variable) []*variable.Variable {
	if _, ok := l.Parameters["w"]; !ok {
		l.inSize = variable.Shape(x[0])[1]

		w := matrix.Randn(l.inSize, l.outSize, l.rnd)
		l.Parameters.Add("w", variable.NewOf(xavier(l.inSize, w)...))
	}

	return []*variable.Variable{
		F.Linear(x[0], l.Parameters["w"], l.Parameters["b"]),
	}
}

func xavier(inSize int, m matrix.Matrix) matrix.Matrix {
	return matrix.MulC(1.0/math.Sqrt(float64(inSize)), m)
}
