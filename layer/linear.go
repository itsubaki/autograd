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
	NoBias bool
	Source rand.Source
}

func Linear(outSize int, opts ...LinearOpts) *LinearT {
	s := rand.NewSource(time.Now().UnixNano())
	if len(opts) != 0 && opts[0].Source != nil {
		s = opts[0].Source
	}

	p := make(Parameters)
	if len(opts) == 0 || !opts[0].NoBias {
		p.Add("b", variable.Zero(1, outSize))
	}

	return &LinearT{
		outSize:    outSize,
		rand:       rand.New(s),
		Parameters: p,
	}
}

type LinearT struct {
	inSize, outSize int
	rand            *rand.Rand
	Parameters
}

func (l *LinearT) First(x ...*variable.Variable) *variable.Variable {
	return l.Forward(x...)[0]
}

func (l *LinearT) Forward(x ...*variable.Variable) []*variable.Variable {
	if _, ok := l.Parameters["w"]; !ok {
		l.inSize = variable.Shape(x[0])[1]

		w := matrix.Randn(l.inSize, l.outSize, l.rand)
		l.Parameters.Add("w", variable.NewOf(xavier(l.inSize, w)...))
	}

	xp := []*variable.Variable{x[0], l.Parameters["w"]}
	if b, ok := l.Parameters["b"]; ok {
		xp = append(xp, b)
	}

	return []*variable.Variable{
		F.Linear(xp...),
	}
}

func xavier(inSize int, m matrix.Matrix) matrix.Matrix {
	return matrix.MulC(1.0/math.Sqrt(float64(inSize)), m)
}
