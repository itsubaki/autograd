package function

import (
	"math"

	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

func Pow(c float64) func(x ...*variable.Variable) []*variable.Variable {
	return (&Function{Forwarder: &PowT{C: c}}).Apply
}

type PowT struct {
	C float64
	x variable.Data
}

func (f *PowT) Forward(x ...variable.Data) []variable.Data {
	f.x = x[0]

	y := vector.Pow(f.x, f.C)
	return []variable.Data{y}
}

func (f *PowT) Backward(gy ...variable.Data) []variable.Data {
	return []variable.Data{vector.F2(f.x, gy[0], dpow(f.C))}
}

func dpow(c float64) func(a, b float64) float64 {
	return func(x, gy float64) float64 {
		return c * math.Pow(x, c-1) * gy
	}
}
