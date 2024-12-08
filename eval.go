package main

import (
	"fmt"
	"math"
	"runtime"
	"time"

	"github.com/dwkim606/test_lattigo/ckks"
)

func set_Variables(batch, raw_in_wid, in_wid, ker_wid int, kind string) (kp_wid, out_batch, logN int, trans bool) {
	N := batch * in_wid * in_wid
	logN = 0
	for ; (1 << logN) < N; logN++ {
	}
	max_kp_wid := in_wid - ((ker_wid - 1) / 2) // max possible size of raw_in_wid

	fmt.Print()
	switch kind {
	case "Conv":
		trans = false
		kp_wid = raw_in_wid
		out_batch = batch
		if kp_wid > max_kp_wid {
			fmt.Println("max raw_in_wid: ", max_kp_wid)
			panic("too large raw_in_wid.")
		}
	case "StrConv", "StrConv_fast", "StrConv_odd":
		trans = false
		kp_wid = 2 * (in_wid/2 - ker_wid/2)
		out_batch = batch
		if kp_wid > max_kp_wid {
			fmt.Println("max raw_in_wid: ", max_kp_wid)
			panic("too large raw_in_wid.")
		}
	case "StrConv_inside":
		trans = false
		kp_wid = (in_wid/2 - ker_wid/2)
		out_batch = batch
	case "TransConv":
		trans = true
		kp_wid = 2 * raw_in_wid
		out_batch = batch / 4
		if kp_wid > max_kp_wid {
			fmt.Println("max raw_in_wid: ", max_kp_wid/2)
			panic("too large raw_in_wid.")
		}
	default:
		panic("Wrong kinds!")
	}

	return
}

func evalConv_BN_BL_test(cont *context, ct_input *ckks.Ciphertext, ker_in, bn_a, bn_b []float64, in_wid, ker_wid, real_ib, real_ob, pos, norm, pad int, trans, printResult bool, ark bool) (ct_res *ckks.Ciphertext) {

	in_size := in_wid * in_wid
	out_size := in_size
	max_batch := cont.N / (2 * in_size)

	// fmt.Println()
	// fmt.Println("===============  (KER) PREPARATION  ===============")
	// fmt.Println()
	start := time.Now()
	max_ker_rs := reshape_ker_BL(ker_in, bn_a, ker_wid, real_ib, real_ob, max_batch, pos, norm, trans)
	scale_exp := cont.params.Scale() * cont.params.Scale()
	if trans {
		scale_exp = cont.params.Scale() * cont.params.Scale() * cont.params.Scale()
	}
	bn_b_slots := make([]complex128, cont.N/2)
	for i, elt := range bn_b {
		for j := 0; j < in_wid-pad; j++ {
			for k := 0; k < in_wid-pad; k++ {
				bn_b_slots[j+k*in_wid+norm*out_size*i] = complex(elt, 0)
			}
		}
	}

	pl_bn_b := ckks.NewPlaintext(cont.params, cont.ECD_LV, scale_exp)
	cont.encoder.EncodeNTT(pl_bn_b, bn_b_slots, cont.logN-1) // encode the BN_B to NTT
	fmt.Printf("Plaintext (kernel) preparation, Done in %s \n", time.Since(start))

	// fmt.Println()
	// fmt.Println("===============  EVALUATION  ===============")
	// fmt.Println()
	start = time.Now()
	ct_inputs_rots := preConv_BL(cont, cont.pack_evaluator, ct_input, in_wid, ker_wid, ark) // have 25 rotated ciphertext, pre rotated
	//fmt.Printf("preConv done in %s \n", time.Since(start))

	var rot_iters int
	if norm*real_ob == max_batch {
		rot_iters = real_ob
	} else {
		rot_iters = max_batch
	}

	for i := 0; i < rot_iters; i++ {
		ct_tmp := postConv_BL(cont, cont.params, cont.encoder, cont.pack_evaluator, ct_inputs_rots, in_wid, ker_wid, norm*i, pad, max_ker_rs)
		if i == 0 {
			ct_res = ct_tmp
		} else {
			if ark {
				rotKey := norm * i * out_size
				rotCipher := ct_tmp.CopyNew()
				for z := 0; z < len(rotArray[0][rotKey]); z++ {
					rotCipher = cont.pack_evaluator.RotateNew(rotCipher, rotArray[0][rotKey][z])
				}
				cont.evaluator.Add(ct_res, rotCipher, ct_res)
			} else {
				cont.evaluator.Add(ct_res, cont.pack_evaluator.RotateNew(ct_tmp, norm*i*out_size), ct_res)
			}
		}
	}

	if ct_res.Scale != scale_exp {
		panic("Different scale between pl_bn_b and ctxt")
	}
	cont.evaluator.Add(ct_res, pl_bn_b, ct_res)
	fmt.Printf("Conv (with BN) Done in %s \n", time.Since(start))

	return ct_res
}

func evalConv_BN(cont *context, ct_input *ckks.Ciphertext, ker_in, bn_a, bn_b []float64, in_wid, ker_wid, real_ib, real_ob, norm int, out_scale float64, trans bool, ark bool) (ct_res *ckks.Ciphertext) {
	max_batch := cont.N / (in_wid * in_wid)

	// fmt.Println()
	// fmt.Println("===============  (KER) PREPARATION  ===============")
	// fmt.Println()
	start := time.Now()
	pl_ker := prep_Ker(cont.params, cont.encoder, ker_in, bn_a, in_wid, ker_wid, real_ib, real_ob, norm, cont.ECD_LV, 0, trans)
	b_coeffs := make([]float64, cont.N)
	for i := range bn_b {
		for j := 0; j < in_wid*in_wid; j++ {
			b_coeffs[norm*i+j*max_batch] = bn_b[i]
		}
	}
	pl_bn_b := ckks.NewPlaintext(cont.params, 0, out_scale)
	cont.encoder.EncodeCoeffs(b_coeffs, pl_bn_b) //bias
	cont.encoder.ToNTT(pl_bn_b)
	t := time.Since(start).Seconds()
	avgTime[2] += t
	fmt.Printf("Plaintext (kernel) preparation, Done in %.6f \n", t)

	// fmt.Println()
	// fmt.Println("===============  EVALUATION  ===============")
	// fmt.Println()
	start = time.Now()

	ct_res = conv_then_pack(cont, cont.params, cont.pack_evaluator, ct_input, pl_ker, cont.pl_idx, max_batch, norm, cont.ECD_LV, out_scale, ark)
	if (pl_bn_b.Scale != ct_res.Scale) || (ct_res.Level() != 0) {
		fmt.Println("plain scale: ", pl_bn_b.Scale)
		fmt.Println("ctxt scale: ", ct_res.Scale)
		fmt.Println("ctxt lv: ", ct_res.Level())
		panic("LV or scale after conv then pack, inconsistent")
	}
	cont.evaluator.Add(ct_res, pl_bn_b, ct_res)
	t = time.Since(start).Seconds()
	avgTime[3] += t
	fmt.Printf("Conv (with BN) Done in %.6f \n", t)

	return ct_res
}

// Eval Conv, BN, relu with Boot
func evalConv_BNRelu_new(cont *context, ct_input *ckks.Ciphertext, ker_in, bn_a, bn_b []float64, alpha, pow float64, in_wid, kp_wid, ker_wid, real_ib, real_ob, norm, pack_pos, step, iter, log_sparse int, kind string, fast_pack, debug bool, ark bool) (ct_res *ckks.Ciphertext) {
	var trans, stride, odd, inside bool
	odd = false
	trans = false
	stride = false
	inside = false
	sparse := false
	in_step := step
	modify_ker := false
	full := false
	switch kind {
	case "Conv_sparse":
		sparse = true
	case "StrConv_sparse":
		modify_ker = true
		sparse = true
		stride = true
	case "StrConv_sparse_full":
		sparse = true
		modify_ker = true
		stride = true
		full = true
	case "Conv_inside":
		inside = true
	case "StrConv_inside":
		in_step = step / 2
		if step%2 != 0 {
			panic("step can not be divided by 2 (for strided conv)")
		}
		inside = true
	case "StrConv", "StrConv_fast":
		stride = true
	case "StrConv_odd":
		stride = true
		odd = true
	case "TransConv":
		trans = true
	case "Conv":
	default:
		panic("No kind!")
	}

	if odd {
		odd_time := time.Now()
		var offset int
		if (in_wid-ker_wid/2)%2 == 0 {
			offset = 0
		} else {
			offset = cont.N / (in_wid * in_wid) * (in_wid + 1)
		}
		xi := make([]float64, cont.N)
		xi[offset] = 1.0
		xi_plain := ckks.NewPlaintext(cont.params, cont.ECD_LV, 1.0)
		cont.encoder.EncodeCoeffs(xi, xi_plain)
		cont.encoder.ToNTT(xi_plain)
		ct_input = cont.evaluator.MulNew(ct_input, xi_plain)
		fmt.Printf("for odd stride, offset time %s \n", time.Since(odd_time))
	}

	var ct_conv *ckks.Ciphertext
	if modify_ker {
		if !full {
			bn_a_0 := make([]float64, real_ib)
			bn_a_1 := make([]float64, real_ib)
			bn_b_0 := make([]float64, real_ib)
			bn_b_1 := make([]float64, real_ib)
			for i := range bn_b_0 {
				bn_a_0[i] = bn_a[2*i]
				bn_a_1[i] = bn_a[2*i+1]
				bn_b_0[i] = bn_b[2*i]
				bn_b_1[i] = bn_b[2*i+1]
			}
			ker_in_0 := make([]float64, len(ker_in)/2)
			ker_in_1 := make([]float64, len(ker_in)/2)
			for k := 0; k < ker_wid*ker_wid; k++ {
				for i := 0; i < real_ib; i++ {
					for j := 0; j < real_ob/2; j++ {
						ker_in_0[k*real_ib*real_ob/2+(i*real_ob/2+j)] = ker_in[k*real_ib*real_ob+(i*real_ob+2*j)]
						ker_in_1[k*real_ib*real_ob/2+(i*real_ob/2+j)] = ker_in[k*real_ib*real_ob+(i*real_ob+2*j+1)]
					}
				}
			}
			ct_result1 := evalConv_BN(cont, ct_input, ker_in_0, bn_a_0, bn_b_0, in_wid, ker_wid, real_ib, real_ob/2, norm/2, math.Exp2(math.Round(math.Log2(float64(cont.params.Q()[0]))-(pow+8))), trans, ark)
			ct_result2 := evalConv_BN(cont, ct_input, ker_in_1, bn_a_1, bn_b_1, in_wid, ker_wid, real_ib, real_ob/2, norm/2, math.Exp2(math.Round(math.Log2(float64(cont.params.Q()[0]))-(pow+8))), trans, ark)

			xi := make([]float64, cont.N)
			offset := norm / 4
			xi[offset] = 1.0
			xi_plain := ckks.NewPlaintext(cont.params, ct_result2.Level(), 1.0)
			cont.encoder.EncodeCoeffs(xi, xi_plain)
			cont.encoder.ToNTT(xi_plain)
			ct_result2 = cont.evaluator.MulNew(ct_result2, xi_plain)

			ct_conv = cont.evaluator.AddNew(ct_result1, ct_result2)
			max_batch := cont.N / (in_wid * in_wid)

			for i := range xi {
				xi[i] = 0.0
			}
			if (in_wid-ker_wid/2)%2 != 0 {
				xi[0] = 1.0
			} else {
				offset = cont.N - (max_batch)*(in_wid+1)
				xi[offset] = -1.0
			}
			xi_plain = ckks.NewPlaintext(cont.params, ct_conv.Level(), 1.0)
			cont.encoder.EncodeCoeffs(xi, xi_plain)
			cont.encoder.ToNTT(xi_plain)
			ct_conv = cont.evaluator.MulNew(ct_conv, xi_plain)
		} else {
			ct_conv = evalConv_BN(cont, ct_input, ker_in, bn_a, bn_b, in_wid, ker_wid, real_ib, real_ob, norm, math.Exp2(math.Round(math.Log2(float64(cont.params.Q()[0]))-(pow+8))), trans, ark)
			max_batch := cont.N / (in_wid * in_wid)
			xi := make([]float64, cont.N)
			for i := range xi {
				xi[i] = 0.0
			}
			var offset int
			if (in_wid-ker_wid/2)%2 != 0 {
				xi[0] = 1.0
			} else {
				offset = cont.N - (max_batch)*(in_wid+1)
				xi[offset] = -1.0
			}
			xi_plain := ckks.NewPlaintext(cont.params, ct_conv.Level(), 1.0)
			cont.encoder.EncodeCoeffs(xi, xi_plain)
			cont.encoder.ToNTT(xi_plain)
			ct_conv = cont.evaluator.MulNew(ct_conv, xi_plain)
		}
	} else {
		if inside {
			new_ker_wid := ker_wid*in_step - in_step + 1
			new_ker_in := make([]float64, len(ker_in)*new_ker_wid*new_ker_wid/(ker_wid*ker_wid))

			for i := 0; i < ker_wid; i++ {
				for j := 0; j < ker_wid; j++ {
					for ib := 0; ib < real_ib; ib++ {
						for ob := 0; ob < real_ob; ob++ {
							new_ker_in[in_step*i*new_ker_wid*real_ib*real_ob+(in_step*j)*real_ib*real_ob+ib*real_ob+ob] = ker_in[i*ker_wid*real_ib*real_ob+j*real_ib*real_ob+ib*real_ob+ob]
						}
					}
				}
			}
			ct_conv = evalConv_BN(cont, ct_input, new_ker_in, bn_a, bn_b, in_wid, new_ker_wid, real_ib, real_ob, norm, math.Exp2(math.Round(math.Log2(float64(cont.params.Q()[0]))-(pow+8))), trans, ark)
		} else {
			ct_conv = evalConv_BN(cont, ct_input, ker_in, bn_a, bn_b, in_wid, ker_wid, real_ib, real_ob, norm, math.Exp2(math.Round(math.Log2(float64(cont.params.Q()[0]))-(pow+8))), trans, ark)
		}
	}

	ct_conv.Scale = ct_conv.Scale * math.Pow(2, pow)
	fmt.Println("Bootstrapping... Ours (until CtoS):")
	start := time.Now()
	ct_boots := make([]*ckks.Ciphertext, 2)

	switch log_sparse {
	case 0:
		ct_boots[0], ct_boots[1], _ = cont.btp.BootstrappConv_CtoS(ct_conv)
	case 1:
		ct_boots[0], ct_boots[1], _ = cont.btp2.BootstrappConv_CtoS(ct_conv)
	case 2:
		ct_boots[0], ct_boots[1], _ = cont.btp3.BootstrappConv_CtoS(ct_conv)
	case 3:
		ct_boots[0], ct_boots[1], _ = cont.btp4.BootstrappConv_CtoS(ct_conv)
	case 4:
		ct_boots[0], ct_boots[1], _ = cont.btp5.BootstrappConv_CtoS(ct_conv)

	default:
		panic("No cases for log_sparse")
	}
	t := time.Since(start).Seconds()
	avgTime[4] += t
	runtime.GC()

	start = time.Now()
	for ul := 0; ul < iter; ul++ {
		if ct_boots[ul] != nil {
			ct_boots[ul] = evalReLU(cont.params, cont.evaluator, ct_boots[ul], alpha)
			cont.evaluator.MulByPow2(ct_boots[ul], int(pow), ct_boots[ul])
		}
	}
	t = time.Since(start).Seconds()
	avgTime[5] += t
	fmt.Printf("ReLU Done in %.6f \n", t)

	start = time.Now()
	ct_keep := make([]*ckks.Ciphertext, iter)
	for ul := 0; ul < iter; ul++ {
		if trans {
			ct_keep[ul] = ext_ctxt(cont.evaluator, cont.encoder, ct_boots[ul], cont.r_idx[in_wid][ul], cont.params, ark)
		} else if stride {
			if sparse {
				if ct_boots[ul] != nil {
					if ul == 0 {
						ct_keep[ul] = ext_double_ctxt(cont, cont.decryptor, cont.evaluator, cont.encoder, ct_boots[ul], cont.m_idx[in_wid][pack_pos], cont.r_idx[in_wid][pack_pos], cont.params, ark)
					} else {
						ct_keep[ul] = ext_double_ctxt(cont, cont.decryptor, cont.evaluator, cont.encoder, ct_boots[ul], cont.m_idx_l[in_wid][pack_pos], cont.r_idx_l[in_wid][pack_pos], cont.params, ark)
					}
				} else {
					ct_keep[ul] = nil
				}
			} else {
				if fast_pack {
					if ul == 0 {
						ct_keep[ul] = ext_double_ctxt(cont, cont.decryptor, cont.evaluator, cont.encoder, ct_boots[ul], cont.m_idx[in_wid][pack_pos], cont.r_idx[in_wid][pack_pos], cont.params, ark)
					} else {
						ct_keep[ul] = ext_double_ctxt(cont, cont.decryptor, cont.evaluator, cont.encoder, ct_boots[ul], cont.m_idx_l[in_wid][pack_pos], cont.r_idx_l[in_wid][pack_pos], cont.params, ark)
					}
				} else {
					if ul == 0 {
						ct_keep[ul] = ext_ctxt(cont.evaluator, cont.encoder, ct_boots[ul], cont.r_idx[in_wid][pack_pos], cont.params, ark)
					} else {
						ct_keep[ul] = ext_ctxt(cont.evaluator, cont.encoder, ct_boots[ul], cont.r_idx_l[in_wid][pack_pos], cont.params, ark)
					}
				}
			}
		} else if inside {
			if ct_boots[ul] != nil {
				if sparse {
					ct_keep[ul] = keep_ctxt(cont.params, cont.evaluator, cont.encoder, ct_boots[ul], cont.ext_idx[in_wid][ul])
				} else {
					ct_keep[ul] = keep_ctxt(cont.params, cont.evaluator, cont.encoder, ct_boots[ul], cont.ext_idx[step][ul])
				}
			} else {
				ct_keep[ul] = nil
			}
		} else {
			if ct_boots[ul] != nil {
				ct_keep[ul] = keep_ctxt(cont.params, cont.evaluator, cont.encoder, ct_boots[ul], cont.ext_idx[in_wid][ul])
			} else {
				ct_keep[ul] = nil
			}
		}
	}

	if iter == 1 {
		ct_boots[1] = nil
		ct_res = cont.btp.BootstrappConv_StoC(ct_keep[0], ct_boots[1])
		if log_sparse != 0 {
			panic("we didn't implement this case")
		}
	} else {
		switch log_sparse {
		case 0:
			ct_res = cont.btp.BootstrappConv_StoC(ct_keep[0], ct_keep[1])
		case 1:
			ct_res = cont.btp2.BootstrappConv_StoC(ct_keep[0], ct_keep[1])
		case 2:
			ct_res = cont.btp3.BootstrappConv_StoC(ct_keep[0], ct_keep[1])
		case 3:
			ct_res = cont.btp4.BootstrappConv_StoC(ct_keep[0], ct_keep[1])
		case 4:
			ct_res = cont.btp5.BootstrappConv_StoC(ct_keep[0], ct_keep[1])
		default:
			panic("No cases for log_sparse")
		}
	}

	cont.evaluator.Rescale(ct_res, cont.params.Scale(), ct_res)
	t = time.Since(start).Seconds()
	avgTime[4] += t
	fmt.Printf("Boot (StoC) Done in %.6f \n", t)
	runtime.GC()

	return ct_res
}
