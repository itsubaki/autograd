package function

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

type Forwarder interface {
	Forward(x ...*variable.Variable) []*variable.Variable
	Backward(gy ...*variable.Variable) []*variable.Variable
}

type Function struct {
	in, out []*variable.Variable
	gen     int
	Forwarder
}

func (f *Function) Input() []*variable.Variable {
	return f.in
}

func (f *Function) Output() []*variable.Variable {
	return f.out
}

func (f *Function) Generation() int {
	return f.gen
}

// ApplyAndFirst applies the function and returns the first output variable.
func (f *Function) ApplyAndFirst(x ...*variable.Variable) *variable.Variable {
	return f.Apply(x...)[0]
}

// Apply applies the function
func (f *Function) Apply(x ...*variable.Variable) []*variable.Variable {
	f.gen = maxgen(x...)

	y := f.Forward(x...)
	for i := range y {
		y[i].SetCreator(f)
	}

	f.in, f.out = x, y
	return f.out
}

func (f Function) String() string {
	return fmt.Sprintf("%T%v", f.Forwarder, f.in)
}

func maxgen(x ...*variable.Variable) int {
	var max int
	for _, v := range x {
		if max < v.Generation {
			max = v.Generation
		}
	}

	return max
}
