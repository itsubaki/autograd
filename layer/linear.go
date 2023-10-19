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
	outSize int
	rand    *rand.Rand
	Parameters
}

func (l *LinearT) First(x ...*variable.Variable) *variable.Variable {
	return l.Forward(x...)[0]
}

func (l *LinearT) Forward(x ...*variable.Variable) []*variable.Variable {
	l.initw(x[0])

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

func (l *LinearT) initw(x *variable.Variable) {
	if _, ok := l.Parameters["w"]; ok {
		return
	}

	inSize := variable.Shape(x)[1]
	w := matrix.Randn(inSize, l.outSize, l.rand)
	l.Parameters.Add("w", variable.NewOf(xavier(inSize, w)...))
}

func xavier(inSize int, m matrix.Matrix) matrix.Matrix {
	return matrix.MulC(1.0/math.Sqrt(float64(inSize)), m)
}
