package hook

import (
	"math"

	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/tensor"
)

func ClipGrad(max float64) func(params []layer.Parameter) {
	return func(params []layer.Parameter) {
		var total float64
		for _, p := range params {
			total += tensor.Sum(tensor.Pow(2, p.Grad.Data)).At()
		}

		rate := max / (math.Sqrt(total) + 1e-6)
		if rate >= 1 {
			return
		}

		for _, p := range params {
			p.Grad.Data = tensor.MulC(rate, p.Grad.Data)
		}
	}
}
