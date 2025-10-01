package optimizer

import (
	"math"

	"github.com/itsubaki/autograd/tensor"
)

type AdamW struct {
	Adam
	WeightDecay float64
}

func (o *AdamW) Update(model Model) {
	o.Adam.update(model, func(lr float64, data, ms, vs *tensor.Tensor[float64]) *tensor.Tensor[float64] {
		update := tensor.F2(ms, vs, func(m, v float64) float64 {
			return lr * m / (math.Sqrt(v) + 1e-8)
		})

		decay := tensor.F(data, func(w float64) float64 {
			return lr * o.WeightDecay * w
		})

		return tensor.Add(update, decay)
	})
}
