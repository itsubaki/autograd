package function

import (
	"fmt"
	"math"

	"github.com/itsubaki/autograd/variable"
)

func Square(x *variable.Variable) *variable.Variable {
	return (&Function{Forwarder: &SquareT{}}).Apply(x)
}

type SquareT struct {
	x variable.Data
}

func (f *SquareT) Forward(x variable.Data) variable.Data {
	f.x = x

	y := variable.NewData(len(x))
	for i := 0; i < len(x); i++ {
		y[i] = math.Pow(x[i], 2)
	}

	return y
}

func (f *SquareT) Backward(gy variable.Data) variable.Data {
	grad := variable.NewData(len(f.x))
	for i := 0; i < len(f.x); i++ {
		grad[i] = 2 * f.x[i] * gy[i] // 2x * gy
	}

	return grad
}

func (f SquareT) String() string {
	return fmt.Sprintf("%T(%v)", f, f.x)
}
