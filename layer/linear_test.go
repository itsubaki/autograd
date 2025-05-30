package layer_test

import (
	"fmt"

	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/rand"
	"github.com/itsubaki/autograd/variable"
)

func ExampleLinear() {
	l := L.Linear(5, L.WithSource(rand.Const()))

	x := variable.New(1, 2, 3)
	y := l.Forward(x)
	fmt.Printf("%.4f\n", y[0].Data.Data)

	for _, v := range l.Params() {
		fmt.Println(v)
	}

	// Unordered output:
	// [-3.7536 -1.7199 0.8735 -0.0434 1.0512]
	// b([0 0 0 0 0])
	// w([[0.32708975344564756 -0.35356774308295924 0.34057587091902425 -0.21236342053185778 0.630441958972765] [-0.25624794608861706 -0.1448844842619606 0.4946976700923148 0.41899488204992313 0.49172779552683027] [-1.1894036897019482 -0.3588555362697145 -0.15215179225530268 -0.2230241557829546 -0.18756042151287408]])
}

func ExampleLinear_inSize() {
	l := L.Linear(5,
		L.WithSource(rand.Const()),
		L.WithInSize(3),
	)

	x := variable.New(1, 2, 3)
	y := l.Forward(x)
	fmt.Printf("%.4f\n", y[0].Data.Data)

	for _, v := range l.Params() {
		fmt.Println(v)
	}

	// Unordered output:
	// [-3.7536 -1.7199 0.8735 -0.0434 1.0512]
	// b([0 0 0 0 0])
	// w([[0.32708975344564756 -0.35356774308295924 0.34057587091902425 -0.21236342053185778 0.630441958972765] [-0.25624794608861706 -0.1448844842619606 0.4946976700923148 0.41899488204992313 0.49172779552683027] [-1.1894036897019482 -0.3588555362697145 -0.15215179225530268 -0.2230241557829546 -0.18756042151287408]])
}

func ExampleLinear_nobias() {
	l := L.Linear(5, L.WithNoBias())

	x := variable.New(1, 2, 3)
	l.Forward(x)

	for _, v := range l.Params() {
		fmt.Println(v.Name)
	}

	// Output:
	// w
}

func ExampleLinear_backward() {
	l := L.Linear(5)

	x := variable.New(1, 2, 3)
	y := l.Forward(x)
	y[0].Backward()

	for _, v := range l.Params() {
		fmt.Println(v.Name, v.Grad)
	}

	y = l.Forward(variable.New(1, 2, 3))
	y[0].Backward()

	for _, v := range l.Params() {
		fmt.Println(v.Name, v.Grad)
	}

	// Unordered output:
	// b variable([1 1 1 1 1])
	// w variable([[1 1 1 1 1] [2 2 2 2 2] [3 3 3 3 3]])
	// b variable([2 2 2 2 2])
	// w variable([[2 2 2 2 2] [4 4 4 4 4] [6 6 6 6 6]])
}
