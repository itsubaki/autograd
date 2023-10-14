package layer

import (
	"math"
	"math/rand"
	"time"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"
)

func Linear(outSize int, s ...rand.Source) *Layer {
	if len(s) == 0 {
		s = append(s, rand.NewSource(time.Now().UnixNano()))
	}

	f := &LinearT{
		params:  make(map[string]Parameter),
		outSize: outSize,
		rnd:     rand.New(s[0]),
	}

	b := variable.Zero(1, outSize)
	b.Name = "b"
	f.params[b.Name] = b
	return &Layer{Forwarder: f}
}

type LinearT struct {
	params          map[string]Parameter
	inSize, outSize int
	rnd             *rand.Rand
}

func (l *LinearT) Forward(x ...*variable.Variable) []*variable.Variable {
	if _, ok := l.params["w"]; !ok {
		l.inSize = variable.Shape(x[0])[1]

		m := matrix.Randn(l.inSize, l.outSize, l.rnd)
		w := variable.NewOf(xavier(l.inSize, m)...)
		w.Name = "w"
		l.params[w.Name] = w
	}

	return []*variable.Variable{
		F.Linear(x[0], l.params["w"], l.params["b"]),
	}
}

func (l *LinearT) Params() map[string]Parameter {
	return l.params
}

func xavier(inSize int, m matrix.Matrix) matrix.Matrix {
	return matrix.MulC(1.0/math.Sqrt(float64(inSize)), m)
}
