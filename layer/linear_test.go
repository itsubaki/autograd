package layer_test

import (
	"fmt"
	"math/rand"

	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

func ExampleLinear() {
	s := rand.NewSource(1)
	l := L.Linear(5, s)

	x := variable.New(1, 2, 3)
	y := l.Apply(x)
	fmt.Printf("%.4f\n", y[0].Data)

	for _, v := range l.Params() {
		fmt.Println(v)
	}

	// Unordered output:
	// [[2.7150 1.5622 3.0911 1.3887 2.2476]]
	// b([0 0 0 0 0])
	// w([[-0.7123106159510769 -0.07294676931545414 -0.300796355901605 1.3196605478957266 0.1863716994911208] [0.34067550733567553 0.09168769154026127 0.571116089650993 -0.42220644624386905 0.3962821310071055] [0.9153334043970172 0.48393840455368875 0.7498861129486725 0.30447051019251964 0.42287554302900365]])
}

func ExampleLinear_backward() {
	l := L.Linear(5)

	x := variable.New(1, 2, 3)
	y := l.Apply(x)
	y[0].Backward()

	for _, v := range l.Params() {
		fmt.Println(v.Name, v.Grad)
	}

	y = l.Apply(variable.New(1, 2, 3))
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

func ExampleLinear_cleargrads() {
	l := L.Linear(5)

	x := variable.New(1, 2, 3)
	y := l.Apply(x)
	y[0].Backward()

	l.Cleargrads()
	for _, v := range l.Params() {
		fmt.Println(v.Name, v.Grad)
	}

	// Unordered output:
	// b <nil>
	// w <nil>
}
