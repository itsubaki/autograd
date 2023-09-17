package function

import (
	"math"

	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

func Sin(x ...*variable.Variable) *variable.Variable {
	return (&Function{Forwarder: &SinT{}}).Apply(x...)[0]
}

type SinT struct {
	x variable.Data
}

func (f *SinT) Forward(x ...variable.Data) []variable.Data {
	f.x = x[0]

	y := vector.F(f.x, sin)
	return []variable.Data{y}
}

func (f *SinT) Backward(gy ...variable.Data) []variable.Data {
	return []variable.Data{vector.F2(f.x, gy[0], dsin)}
}

func sin(a float64) float64 { return math.Sin(a) }

func dsin(x, gy float64) float64 { return cos(x, gy) }

func cos(x, gy float64) float64 { return math.Cos(x) * gy }
