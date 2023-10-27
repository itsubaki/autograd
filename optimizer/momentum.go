package optimizer

import "github.com/itsubaki/autograd/matrix"

type Momentum struct {
	LearningRate float64
	Momentum     float64
	Hooks        []Hook
	vs           map[string]matrix.Matrix
}

// Update updates the parameters of the model.
func (o *Momentum) Update(model Model) {
	params := Params(model)
	for _, h := range o.Hooks {
		h(params)
	}

	if len(o.vs) == 0 {
		o.vs = make(map[string]matrix.Matrix)
	}

	for _, p := range params {
		if _, ok := o.vs[id(p)]; !ok {
			o.vs[id(p)] = matrix.ZeroLike(p.Data)
		}

		v := o.vs[id(p)]
		v = matrix.F2(v, p.Grad.Data, func(v, grad float64) float64 { return o.Momentum*v - o.LearningRate*grad }) // v = momentum * v - learning_rate * grad
		o.vs[id(p)] = v

		p.Data = matrix.Add(p.Data, v)
	}
}
