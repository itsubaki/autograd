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

func OneLike(v *Variable) *Variable {
	w := NewData(len(v.Data))
	for i := 0; i < len(v.Data); i++ {
		w[i] = 1
	}

	return &Variable{Data: w}
}

func Copy(v *Variable) *Variable {
	w := NewData(len(v.Data))
	copy(w, v.Data)

	return &Variable{Data: w}
}

func (v *Variable) Backward() {
	if len(v.Grad) == 0 {
		v.Grad = OneLike(v).Data
	}

	fs := []Function{v.Creator}
	for {
		if len(fs) == 0 || fs[0] == nil {
			break
		}

		// pop
		f := fs[len(fs)-1]
		fs = fs[:len(fs)-1]

		// backward
		x, y := f.Input(), f.Output()
		x.Grad = f.Backward(y.Grad)

		if x.Creator != nil {
			fs = append(fs, x.Creator)
		}
	}
}

func (v Variable) String() string {
	return fmt.Sprintf("variable(%v)", v.Data)
}

func gxs(v []Variable) []Data {
	gxs := make([]Data, len(v))
	for i := 0; i < len(v); i++ {
		gxs[i] = v[i].Grad
	}

	return gxs
}
