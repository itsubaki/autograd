package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleSub() {
	v := variable.New(2, 3)
	w := variable.New(3, 4)
	y := F.Sub(v, w)
	y.Backward()

	fmt.Println(y)
	fmt.Println(v.Grad)
	fmt.Println(w.Grad)

	// Output:
	// variable[-1 -1]
	// variable[1 1]
	// variable[-1 -1]
}
