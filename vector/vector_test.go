package vector_test

import (
	"fmt"

	"github.com/itsubaki/autograd/vector"
)

func ExampleZeroLike() {
	v := []float64{1, 2, 3, 4, 5}
	fmt.Println(vector.ZeroLike(v))

	// Output:
	// [0 0 0 0 0]
}

func ExampleOneLike() {
	v := []float64{1, 2, 3, 4, 5}
	fmt.Println(vector.OneLike(v))

	// Output:
	// [1 1 1 1 1]
}

func ExampleConst() {
	fmt.Println(vector.Const(1))

	// Output:
	// [1]
}

func ExampleShape() {
	fmt.Println(vector.Shape([]float64{1, 2, 3, 4, 5}))

	// Output:
	// [1 5]
}

func ExampleInt() {
	fmt.Println(vector.Int([]float64{1.0, 2.0, 3.0, 4.0, 5.0}))

	// Output:
	// [1 2 3 4 5]
}

func ExampleSumTo() {
	v := []float64{1, 2, 3, 4, 5}
	fmt.Println(vector.SumTo(nil, v))

	// Output:
	// 15
}

func ExampleSum() {
	v := []float64{1, 2, 3, 4, 5}
	fmt.Println(vector.Sum(v))

	// Output:
	// 15
}

func ExampleBroadcastTo() {
	v := []float64{1}
	fmt.Println(vector.BroadcastTo([]int{1, 10}, v))

	// Output:
	// [1 1 1 1 1 1 1 1 1 1]
}

func ExampleBroadcast() {
	v := []float64{1}
	w := []float64{5, 6, 7, 8, 9}
	fmt.Println(vector.Broadcast(v, w))
	fmt.Println(vector.Broadcast(w, v))
	fmt.Println(vector.Broadcast(w, w))

	// Output:
	// [1 1 1 1 1] [5 6 7 8 9]
	// [5 6 7 8 9] [1 1 1 1 1]
	// [5 6 7 8 9] [5 6 7 8 9]
}

func ExampleTranspose() {
	fmt.Println(vector.Transpose([]float64{1, 2, 3, 4, 5}))

	// Output:
	// [[1] [2] [3] [4] [5]]
}

func ExampleF() {
	v := []float64{1, 2, 3, 4, 5}
	fmt.Println(vector.F(v, func(a float64) float64 { return a * a }))

	fmt.Println(vector.AddC(1, v))
	fmt.Println(vector.SubC(1, v))
	fmt.Println(vector.MulC(2, v))
	fmt.Println(vector.Exp(v))
	fmt.Println(vector.Log(v))
	fmt.Println(vector.Sin(v))
	fmt.Println(vector.Cos(v))
	fmt.Println(vector.Tanh(v))
	fmt.Println(vector.Pow(2.0, v))

	// Output:
	// [1 4 9 16 25]
	// [2 3 4 5 6]
	// [0 -1 -2 -3 -4]
	// [2 4 6 8 10]
	// [2.718281828459045 7.38905609893065 20.085536923187668 54.598150033144236 148.4131591025766]
	// [0 0.6931471805599453 1.0986122886681096 1.3862943611198906 1.6094379124341003]
	// [0.8414709848078965 0.9092974268256816 0.1411200080598672 -0.7568024953079282 -0.9589242746631385]
	// [0.5403023058681398 -0.4161468365471424 -0.9899924966004454 -0.6536436208636119 0.2836621854632263]
	// [0.7615941559557649 0.9640275800758169 0.9950547536867305 0.999329299739067 0.9999092042625951]
	// [1 4 9 16 25]
}

func ExampleF2() {
	v := []float64{1, 2, 3, 4, 5}
	w := []float64{6, 7, 8, 9, 10}
	fmt.Println(vector.F2(v, w, func(a, b float64) float64 { return a * b }))

	fmt.Println(vector.Add(v, w))
	fmt.Println(vector.Sub(v, w))
	fmt.Println(vector.Mul(v, w))
	fmt.Println(vector.Div(v, w))

	// Output:
	// [6 14 24 36 50]
	// [7 9 11 13 15]
	// [-5 -5 -5 -5 -5]
	// [6 14 24 36 50]
	// [0.16666666666666666 0.2857142857142857 0.375 0.4444444444444444 0.5]
}

func ExampleEquals() {
	fmt.Println(vector.Equals([]int{1, 2}, []int{1, 2}))
	fmt.Println(vector.Equals([]int{1, 2}, []int{2, 1}))
	fmt.Println(vector.Equals([]int{1, 2}, []int{1}))

	// Output:
	// true
	// false
	// false
}
