package optimizer

import (
	"math"

	"github.com/itsubaki/autograd/tensor"
)

// Adam is an optimizer that uses the Adam algorithm.
type Adam struct {
	Alpha float64
	Beta1 float64
	Beta2 float64
	Hook  []Hook
	Iter  int
	Ms    map[string]*tensor.Tensor[float64]
	Vs    map[string]*tensor.Tensor[float64]
}

// Update updates the parameters of the model.
func (o *Adam) Update(model Model) {
	if len(o.Ms) == 0 {
		o.Ms = make(map[string]*tensor.Tensor[float64])
		o.Vs = make(map[string]*tensor.Tensor[float64])
	}

	o.Iter++
	fix1 := 1.0 - math.Pow(o.Beta1, float64(o.Iter))
	fix2 := 1.0 - math.Pow(o.Beta2, float64(o.Iter))
	lr := o.Alpha * math.Sqrt(fix2) / fix1

	params := Params(model, o.Hook)
	for name, p := range params {
		if _, ok := o.Ms[name]; !ok {
			o.Ms[name] = tensor.ZeroLike(p.Data)
			o.Vs[name] = tensor.ZeroLike(p.Data)
		}

		o.Ms[name] = tensor.F2(o.Ms[name], p.Grad.Data, func(m, grad float64) float64 {
			return m + (1-o.Beta1)*(grad-m)
		})

		o.Vs[name] = tensor.F2(o.Vs[name], p.Grad.Data, func(v, grad float64) float64 {
			return v + (1-o.Beta2)*(grad*grad-v)
		})

		// update function
		update := tensor.F2(o.Ms[name], o.Vs[name], func(m, v float64) float64 {
			return lr * m / (math.Sqrt(v) + 1e-8)
		})

		// param = param - (lr * m / (sqrt(v) + 1e-8))
		p.Data = tensor.Sub(p.Data, update)
	}
}
