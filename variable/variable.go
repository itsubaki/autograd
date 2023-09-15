package variable

import "fmt"

type Data = []float64

func NewData(n int) Data {
	return make(Data, n)
}

type Function interface {
	Input() *Variable
	Output() *Variable
	Backward(gy Data) Data
}

type Variable struct {
	Data, Grad Data
	Creator    Function
}

func New(v ...float64) *Variable {
	return &Variable{Data: v}
}

func Copy(v *Variable) *Variable {
	w := NewData(len(v.Data))
	copy(w, v.Data)

	return &Variable{Data: w}
}

func OneLike(x Data) Data {
	w := NewData(len(x))
	for i := 0; i < len(x); i++ {
		w[i] = 1
	}

	return w
}

func (v *Variable) Backward() {
	f := v.Creator
	if f == nil {
		return
	}

	if len(v.Grad) == 0 {
		v.Grad = OneLike(v.Data)
	}

	x := f.Input()
	x.Grad = f.Backward(v.Grad)
	x.Backward()
}

func (v Variable) String() string {
	return fmt.Sprintf("variable(%v)", v.Data)
}
