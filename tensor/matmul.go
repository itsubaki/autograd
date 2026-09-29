//go:build !darwin

package tensor

func matmul(a, b, o []float32, m, k, n int) {
	for i := range m {
		ai := i * k
		oi := i * n

		for p := range k {
			aip := a[ai+p]
			bp := p * n

			for j := range n {
				o[oi+j] += aip * b[bp+j]
			}
		}
	}
}
