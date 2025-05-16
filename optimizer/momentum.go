package optimizer

import (
	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"
)

type Momentum struct {
	LearningRate float64
	Momentum     float64
	Hook         []Hook
	vs           map[*variable.Variable]*matrix.Matrix
}

// Update updates the parameters of the model.
func (o *Momentum) Update(model Model) {
	params := Params(model, o.Hook)

	if len(o.vs) == 0 {
		o.vs = make(map[*variable.Variable]*matrix.Matrix)
	}

	for _, p := range params {
		if _, ok := o.vs[p]; !ok {
			o.vs[p] = matrix.ZeroLike(p.Data)
		}

		// param = param + (momentum * v - lr * grad)
		o.vs[p] = matrix.F2(o.vs[p], p.Grad.Data, momentum(o.Momentum, o.LearningRate))
		p.Data = matrix.Add(p.Data, o.vs[p])
	}
}

func momentum(momentum, lr float64) func(v, grad float64) float64 {
	return func(v, grad float64) float64 { return momentum*v - lr*grad }
}
