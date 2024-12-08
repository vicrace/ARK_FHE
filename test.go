package main

import (
	"fmt"
	"runtime"
	"strconv"
	"time"

	"github.com/dwkim606/test_lattigo/ckks"
)

func testConv_in(in_batch, in_wid, ker_wid, total_test_num int, boot bool, ark bool) {
	kind := "Conv"
	printResult := false
	raw_in_batch := in_batch
	raw_in_wid := in_wid - ker_wid/2
	norm := in_batch / raw_in_batch
	test_dir := "test_conv_data/"
	pow := 4.0

	kp_wid, out_batch, logN, trans := set_Variables(in_batch, raw_in_wid, in_wid, ker_wid, kind)
	raw_out_batch := out_batch / norm

	cont := newContext(logN, ker_wid, []int{in_wid}, []int{kp_wid}, boot, kind, ark)
	printMemUsage("Rotation Key End")

	// fmt.Println("vec size: log2 = ", cont.logN)
	// fmt.Println("raw input width: ", raw_in_wid)
	// fmt.Println("kernel width: ", ker_wid)
	// fmt.Println("num raw batches in & out: ", raw_in_batch, ", ", raw_out_batch)
	// fmt.Println("in_wed: ", in_wid, " ker_wid: ", ker_wid, " total test: ", total_test_num)

	for test_iter := 0; test_iter < total_test_num; test_iter++ {
		fmt.Println("\n", test_iter+1, "-th iter...start")
		raw_input := readTxt(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_in_"+strconv.Itoa(test_iter)+".csv", raw_in_wid*raw_in_wid*raw_in_batch)
		ker_in := readTxt(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_ker_"+strconv.Itoa(test_iter)+".csv", raw_in_batch*raw_out_batch*ker_wid*ker_wid)
		bn_a := readTxt(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_bna_"+strconv.Itoa(test_iter)+".csv", raw_out_batch)
		bn_b := readTxt(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_bnb_"+strconv.Itoa(test_iter)+".csv", raw_out_batch)

		// fmt.Println(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_in_"+strconv.Itoa(test_iter)+".csv", raw_in_wid*raw_in_wid*in_batch)
		// fmt.Println(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_ker_"+strconv.Itoa(test_iter)+".csv", in_batch*in_batch*ker_wid*ker_wid)
		// fmt.Println(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_bna_"+strconv.Itoa(test_iter)+".csv", in_batch)
		// fmt.Println(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_bnb_"+strconv.Itoa(test_iter)+".csv", in_batch)

		input := prep_Input(raw_input, raw_in_wid, in_wid, cont.N, norm, trans, printResult)
		//start := time.Now()
		plain_tmp := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
		cont.encoder.EncodeCoeffs(input, plain_tmp)
		ctxt_input := cont.encryptor.EncryptNew(plain_tmp)
		//fmt.Printf("Encryption done in %s \n", time.Since(start))

		// Kernel Prep & Conv (+BN) Evaluation
		var ct_result *ckks.Ciphertext
		if boot {
			ct_result = evalConv_BNRelu_new(cont, ctxt_input, ker_in, bn_a, bn_b, 0.0, pow, in_wid, kp_wid, ker_wid, raw_in_batch, raw_out_batch, norm, 0, 0, 2, 0, kind, false, false, ark)
		} else {
			ct_result = evalConv_BN(cont, ctxt_input, ker_in, bn_a, bn_b, in_wid, ker_wid, raw_in_batch, raw_out_batch, norm, float64(1<<30), trans, ark)
		}

		//start = time.Now()
		cont.decryptor.Decrypt(ct_result, plain_tmp)
		cfs_tmp := cont.encoder.DecodeCoeffs(plain_tmp)
		//fmt.Printf("Decryption Done in %s \n", time.Since(start))

		var real_out []float64
		if boot {
			real_out = readTxt(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_reluout_"+strconv.Itoa(test_iter)+".csv", raw_in_wid*raw_in_wid*raw_in_batch)
		} else {
			real_out = readTxt(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_out_"+strconv.Itoa(test_iter)+".csv", raw_in_wid*raw_in_wid*raw_in_batch)
		}
		test_out := post_process(cfs_tmp, raw_in_wid, in_wid, real_out)

		printDebugCfsPlain(test_out, real_out)
	}

}

func testResNet_crop_sparse(st, end, ker_wid, depth int, debug, cf100 bool, ark bool) {
	ker_name := "ker" + strconv.Itoa(ker_wid)
	weight_dir := "Resnet_weights/weights_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid1/"
	out_dir := "Resnet_enc_results/results_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid1/"
	fc_out := 10
	init_pow := 6.0
	mid_pow := 6.0
	final_pow := 6.0

	//Reset time measurement
	avgTime[0] = 0
	avgTime[1] = 0
	avgTime[2] = 0
	avgTime[3] = 0
	avgTime[4] = 0
	avgTime[5] = 0
	avgTime[6] = 0
	allocMem = 0

	var num_blcs [3]int
	switch depth {
	case 20:
		num_blcs[0], num_blcs[1], num_blcs[2] = 7, 5, 5
	case 14:
		num_blcs[0], num_blcs[1], num_blcs[2] = 5, 3, 3
	case 8:
		num_blcs[0], num_blcs[1], num_blcs[2] = 3, 1, 1
	default:
		panic("wrong depth (not in 8, 14, 20)!")
	}
	real_batch := []int{16, 32, 64}
	norm := []int{4, 8, 16}
	step := []int{1, 1, 1}

	logN := 16
	alpha := 0.0
	in_wids := []int{32, 16, 8}
	raw_in_wids := []int{32 - ker_wid/2, 16 - ker_wid/2, 8 - ker_wid/2}
	fast_pack := true
	ker_size := ker_wid * ker_wid
	max_batch := make([]int, len(real_batch))
	for i := range max_batch {
		max_batch[i] = (1 << logN) / (in_wids[i] * in_wids[i])
	}

	cont := newContext(logN, ker_wid, in_wids, raw_in_wids, true, "Resnet_crop_sparse", ark)
	printMemUsage("Rotation Key End")

	for iter := st; iter < end; iter++ {
		fmt.Println("\nRunning ", iter, "-th iter... ker size: ", ker_wid)
		image := readTxt("Resnet_plain_data/crop_ker"+strconv.Itoa(ker_wid)+"_d"+strconv.Itoa(depth)+"_wid1/test_image_"+strconv.Itoa(iter)+".csv", in_wids[0]*in_wids[0]*3)

		if cf100 {
			image = readTxt("Resnet_plain_data/cf100_crop_ker"+strconv.Itoa(ker_wid)+"_d"+strconv.Itoa(depth)+"_wid1/test_image_"+strconv.Itoa(iter)+".csv", in_wids[0]*in_wids[0]*3)
		}
		input := make([]float64, cont.N)
		k := 0
		for i := 0; i < in_wids[0]; i++ {
			for j := 0; j < in_wids[0]; j++ {
				for b := 0; b < 3; b++ {
					if (i < raw_in_wids[0]) && (j < raw_in_wids[0]) {
						input[i*in_wids[0]*max_batch[0]+j*max_batch[0]+b*norm[0]] = image[k]
					}
					k++
				}
			}
		}

		enc_start := time.Now()
		pl_input := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale())
		cont.encoder.EncodeCoeffs(input, pl_input)
		ct_input := cont.encryptor.EncryptNew(pl_input)
		//t := time.Since(enc_start).Seconds()
		avgTime[1] += time.Since(enc_start).Seconds()
		//fmt.Printf("Encryption done in %.6f \n", t)

		timings := make([]float64, 6)
		start := time.Now()

		// ResNet Block 1
		pow := init_pow
		ct_layer := ct_input
		for i := 1; i <= num_blcs[0]; i++ {
			bn_a := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-a.csv", real_batch[0])
			bn_b := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-b.csv", real_batch[0])
			ker_in_batch := 3
			if i != 1 {
				ker_in_batch = real_batch[0]
			}
			ker_in := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-conv.csv", ker_in_batch*real_batch[0]*ker_size)
			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in, bn_a, bn_b, alpha, pow, in_wids[0], raw_in_wids[0], ker_wid, ker_in_batch, real_batch[0], norm[0], 0, step[0], 2, 2, "Conv_sparse", fast_pack, debug, ark)
			pow = mid_pow
			fmt.Println("Block1, Layer ", i, "done!")
		}
		fmt.Println("Block1 done.")
		timings[0] = time.Since(start).Seconds()
		start = time.Now()

		ker_in12 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-conv.csv", real_batch[0]*real_batch[1]*ker_size)
		bn_a := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-a.csv", real_batch[1])
		bn_b := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-b.csv", real_batch[1])

		ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in12, bn_a, bn_b, alpha, pow, in_wids[0], raw_in_wids[1], ker_wid, real_batch[0], real_batch[1], norm[1], 0, step[1], 2, 1, "StrConv_sparse", fast_pack, debug, ark)
		fmt.Println("Block1 to 2 done!")
		timings[1] = time.Since(start).Seconds()
		start = time.Now()

		// ResNet Block 2
		for i := 1; i <= num_blcs[1]; i++ {
			bn_a2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-a.csv", real_batch[1])
			bn_b2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-b.csv", real_batch[1])
			ker_in2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-conv.csv", real_batch[1]*real_batch[1]*ker_size)
			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in2, bn_a2, bn_b2, alpha, pow, in_wids[1], raw_in_wids[1], ker_wid, real_batch[1], real_batch[1], norm[1], 0, step[1], 2, 3, "Conv_sparse", fast_pack, debug, ark)
			fmt.Println("Block2, Layer ", i, "done!")
		}
		fmt.Println("Block2 done.")
		timings[2] = time.Since(start).Seconds()
		start = time.Now()

		ker_in23 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-conv.csv", real_batch[1]*real_batch[2]*ker_size)
		bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-a.csv", real_batch[2])
		bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-b.csv", real_batch[2])

		ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in23, bn_a3, bn_b3, alpha, pow, in_wids[1], raw_in_wids[2], ker_wid, real_batch[1], real_batch[2], norm[2], 0, step[2], 2, 2, "StrConv_sparse", fast_pack, debug, ark)
		fmt.Println("Block2 to 3 done!")
		timings[3] = time.Since(start).Seconds()
		start = time.Now()

		// ResNet Block 3
		for i := 1; i <= num_blcs[2]; i++ {
			bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-a.csv", real_batch[2])
			bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-b.csv", real_batch[2])
			ker_in3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-conv.csv", real_batch[2]*real_batch[2]*ker_size)

			if i == num_blcs[2] {
				pow = final_pow
			}
			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in3, bn_a3, bn_b3, alpha, pow, in_wids[2], raw_in_wids[2], ker_wid, real_batch[2], real_batch[2], norm[2], 0, step[2], 2, 4, "Conv_sparse", fast_pack, debug, ark)
			fmt.Println("Block3, Layer ", i, "done!")
		}
		fmt.Println("Block3 done.")
		timings[4] = time.Since(start).Seconds()
		start = time.Now()

		ker_inf_wid := raw_in_wids[2]
		if ker_inf_wid%2 == 0 {
			ker_inf_wid++
		}
		ker_inf := readTxt(weight_dir+"final-fckernel.csv", real_batch[2]*fc_out)

		var ct_result, ct_result2 *ckks.Ciphertext
		if cf100 {
			ker_inf_1 := make([]float64, ker_inf_wid*ker_inf_wid*real_batch[2]*fc_out/2)
			ker_inf_2 := make([]float64, ker_inf_wid*ker_inf_wid*real_batch[2]*fc_out/2)
			for i := 0; i < fc_out/2; i++ {
				for j := 0; j < real_batch[2]; j++ {
					for b := 0; b < ker_inf_wid*ker_inf_wid; b++ {
						ker_inf_1[j*fc_out/2+i+b*real_batch[2]*fc_out/2] = ker_inf[j*fc_out+i]
						ker_inf_2[j*fc_out/2+i+b*real_batch[2]*fc_out/2] = ker_inf[j*fc_out+i+fc_out/2]
					}
				}
			}
			bn_af := make([]float64, fc_out/2)
			for i := range bn_af {
				bn_af[i] = 1.0 / float64(raw_in_wids[2]*raw_in_wids[2]) // for reduce mean on raw_in_wids[2]**2 elements
			}
			bn_bf := readTxt(weight_dir+"final-fcbias.csv", fc_out)
			bn_bf_1 := make([]float64, fc_out/2)
			bn_bf_2 := make([]float64, fc_out/2)
			for i := range bn_bf_1 {
				bn_bf_1[i] = bn_bf[i]
				bn_bf_2[i] = bn_bf[i+fc_out/2]
			}
			ct_result = evalConv_BN(cont, ct_layer, ker_inf_1, bn_af, bn_bf_1, in_wids[2], ker_inf_wid, real_batch[2], fc_out/2, norm[2], float64(1<<30), false, ark)
			ct_result2 = evalConv_BN(cont, ct_layer, ker_inf_2, bn_af, bn_bf_2, in_wids[2], ker_inf_wid, real_batch[2], fc_out/2, norm[2], float64(1<<30), false, ark)
			fmt.Println("Final FC done.")
			timings[5] = time.Since(start).Seconds()
			start = time.Now()
		} else {
			ker_inf_ := make([]float64, ker_inf_wid*ker_inf_wid*real_batch[2]*fc_out)
			for i := range ker_inf {
				for b := 0; b < ker_inf_wid*ker_inf_wid; b++ {
					ker_inf_[i+b*real_batch[2]*fc_out] = ker_inf[i]
				}
			}
			bn_af := make([]float64, fc_out)
			for i := range bn_af {
				bn_af[i] = 1.0 / float64(raw_in_wids[2]*raw_in_wids[2])
			}
			bn_bf := readTxt(weight_dir+"final-fcbias.csv", fc_out)
			ct_result = evalConv_BN(cont, ct_layer, ker_inf_, bn_af, bn_bf, in_wids[2], ker_inf_wid, real_batch[2], fc_out, norm[2], float64(1<<30), false, ark)
			fmt.Println("Final FC done.")
			timings[5] = time.Since(start).Seconds()
			start = time.Now()
		}

		if cf100 {
			cont.decryptor.Decrypt(ct_result, pl_input)
			res_tmp1 := cont.encoder.DecodeCoeffs(pl_input)
			cont.decryptor.Decrypt(ct_result2, pl_input)
			res_tmp2 := cont.encoder.DecodeCoeffs(pl_input)
			t := time.Since(start).Seconds()
			avgTime[6] += t
			res_out := append(prt_mat_one_norm(res_tmp1, max_batch[2], norm[2], ker_inf_wid/2+1, ker_inf_wid/2+1)[:fc_out/2], prt_mat_one_norm(res_tmp2, max_batch[2], norm[2], ker_inf_wid/2+1, ker_inf_wid/2+1)[:fc_out/2]...)
			fmt.Println("\n result: ", res_out)
			writeTxt(out_dir+"class_result_"+ker_name+"_"+strconv.Itoa(iter)+".csv", res_out)
		} else {
			cont.decryptor.Decrypt(ct_result, pl_input)
			res_tmp := cont.encoder.DecodeCoeffs(pl_input)
			t := time.Since(start).Seconds()
			avgTime[6] += t
			res_out := prt_mat_one_norm(res_tmp, max_batch[2], norm[2], ker_inf_wid/2+1, ker_inf_wid/2+1)
			fmt.Println("Want: ", len(res_out), "  ", res_out[:fc_out])
			writeTxt(out_dir+"class_result_"+ker_name+"_"+strconv.Itoa(iter)+".csv", res_out[:fc_out])
		}

		count := iter + 1
		setupavg := avgTime[0]
		encryptevg := avgTime[1] / float64(count)
		kernelavg := avgTime[2] / float64(count)
		convavg := avgTime[3] / float64(count)
		bootavg := avgTime[4] / float64(count)
		reluavg := avgTime[5] / float64(count)
		decryptavg := avgTime[6] / float64(count)

		ot := (convavg + bootavg + reluavg)
		minutes := int(ot) / 60
		seconds := int(ot) % 60

		tt := (setupavg + encryptevg + kernelavg + convavg + bootavg + reluavg + decryptavg)
		minutes1 := int(tt) / 60
		seconds1 := int(tt) % 60
		runtime.GC()

		fmt.Printf("\nSetup Avg : %.6fs,   Encrypt Avg: %.6fs,  Kernel Avg: %.6fs,  Conv Avg: %.6fs,   Boot Avg: %.6fs,   Relu Avg : %.6fs,  Decrypt Avg: %.6fs \n", setupavg, encryptevg, kernelavg, convavg, bootavg, reluavg, decryptavg)
		fmt.Printf("Execution time: %02d:%02d, Total time (include setup, enc & dec): %02d:%02d , Memory: %.2f GB\n", minutes, seconds, minutes1, seconds1, bToGb(allocMem))
	}

}

func testResNet_crop_sparse_wide(st, end, ker_wid, depth, wide_case int, debug, cf100 bool, ark bool) {
	ker_name := "ker" + strconv.Itoa(ker_wid)
	weight_dir := "Resnet_weights/weights_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid" + strconv.Itoa(wide_case) + "/"
	out_dir := "Resnet_enc_results/results_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid" + strconv.Itoa(wide_case) + "/"
	fc_out := 10

	//Reset time measurement
	avgTime[0] = 0
	avgTime[1] = 0
	avgTime[2] = 0
	avgTime[3] = 0
	avgTime[4] = 0
	avgTime[5] = 0
	avgTime[6] = 0
	allocMem = 0

	init_pow := 5.0
	mid_pow := 5.0
	final_pow := 5.0
	if ker_wid == 5 {
		init_pow = 6.0
		mid_pow = 6.0
		final_pow = 6.0
	}

	if cf100 {
		weight_dir = "Resnet_weights/weights_cf100_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid" + strconv.Itoa(wide_case) + "/"
		out_dir = "Resnet_enc_results/results_cf100_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid" + strconv.Itoa(wide_case) + "/"
		fc_out = 100
		final_pow = 7.0
		init_pow = 5.0
		mid_pow = 5.0
		if (ker_wid == 5) && (depth == 8) {
			init_pow = 6.0
			final_pow = 6.0
		}
	}

	init_batch := 16

	var num_blcs [3]int
	switch depth {
	case 20:
		num_blcs[0], num_blcs[1], num_blcs[2] = 7, 5, 5
	case 14:
		num_blcs[0], num_blcs[1], num_blcs[2] = 5, 3, 3
	case 8:
		num_blcs[0], num_blcs[1], num_blcs[2] = 3, 1, 1
	default:
		panic("wrong depth case (not in 8,14,20)!")
	}
	real_batch := []int{32, 64, 128}
	norm := []int{2, 4, 8}
	log_sparse := []int{1, 2, 3}
	step := []int{1, 1, 1}
	kind := "Resnet_crop_sparse_wide2"

	if wide_case == 3 {
		real_batch = []int{48, 96, 192}
		norm = []int{1, 2, 4}
		log_sparse = []int{0, 1, 2}
		kind = "Resnet_crop_sparse_wide3"
	} else if wide_case != 2 {
		panic("wrong wide_case (2 nor 3)!")
	}

	logN := 16
	alpha := 0.0
	in_wids := []int{32, 16, 8}
	raw_in_wids := []int{32 - ker_wid/2, 16 - ker_wid/2, 8 - ker_wid/2}
	fast_pack := true
	ker_size := ker_wid * ker_wid
	max_batch := make([]int, len(real_batch))
	for i := range max_batch {
		max_batch[i] = (1 << logN) / (in_wids[i] * in_wids[i])
	}

	cont := newContext(logN, ker_wid, in_wids, raw_in_wids, true, kind, ark)
	printMemUsage("Rotation Key End")

	for iter := st; iter < end; iter++ {
		fmt.Println("\nRunning ", iter, "-th iter... ker size: ", ker_wid)
		image := readTxt("Resnet_plain_data/crop_ker"+strconv.Itoa(ker_wid)+"_d"+strconv.Itoa(depth)+"_wid"+strconv.Itoa(wide_case)+"/test_image_"+strconv.Itoa(iter)+".csv", in_wids[0]*in_wids[0]*3)

		if cf100 {
			image = readTxt("Resnet_plain_data/cf100_crop_ker"+strconv.Itoa(ker_wid)+"_d"+strconv.Itoa(depth)+"_wid"+strconv.Itoa(wide_case)+"/test_image_"+strconv.Itoa(iter)+".csv", in_wids[0]*in_wids[0]*3)
		}
		input := make([]float64, cont.N)
		k := 0
		for i := 0; i < in_wids[0]; i++ {
			for j := 0; j < in_wids[0]; j++ {
				for b := 0; b < 3; b++ {
					if (i < raw_in_wids[0]) && (j < raw_in_wids[0]) {
						input[i*in_wids[0]*max_batch[0]+j*max_batch[0]+b*norm[0]] = image[k]
					}
					k++
				}
			}
		}

		enc_start := time.Now()
		pl_input := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale())
		cont.encoder.EncodeCoeffs(input, pl_input)
		ct_input := cont.encryptor.EncryptNew(pl_input)
		t := time.Since(enc_start).Seconds()
		avgTime[1] += t
		fmt.Printf("Encryption done in %.6f \n", t)
		enc_start = time.Now()

		timings := make([]float64, 6)
		start := time.Now()

		// ResNet Block 1
		pow := init_pow
		ct_layer := ct_input
		for i := 1; i <= num_blcs[0]; i++ {
			if i == 5 {
				pow = mid_pow
			}
			var bn_batch int
			if i == 1 {
				bn_batch = init_batch
			} else {
				bn_batch = real_batch[0]
			}
			bn_a := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-a.csv", bn_batch)
			bn_b := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-b.csv", bn_batch)

			if i == 1 {
				ker_in := readTxt(weight_dir+"w0-conv.csv", 3*init_batch*ker_size)
				ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in, bn_a, bn_b, alpha, pow, in_wids[0], raw_in_wids[0], ker_wid, 3, init_batch, norm[0], 0, step[0], 2, log_sparse[0], "Conv_sparse", fast_pack, debug, ark)
			} else if i == 2 {
				ker_in := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-conv.csv", init_batch*real_batch[0]*ker_size)
				ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in, bn_a, bn_b, alpha, pow, in_wids[0], raw_in_wids[0], ker_wid, init_batch, real_batch[0], norm[0], 0, step[0], 2, log_sparse[0], "Conv_sparse", fast_pack, debug, ark)
			} else {
				ker_in := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-conv.csv", real_batch[0]*real_batch[0]*ker_size)
				ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in, bn_a, bn_b, alpha, pow, in_wids[0], raw_in_wids[0], ker_wid, real_batch[0], real_batch[0], norm[0], 0, step[0], 2, log_sparse[0], "Conv_sparse", fast_pack, debug, ark)
			}
			fmt.Println("Block1, Layer ", i, "done!")
		}
		fmt.Println("Block1 done.")
		timings[0] = time.Since(start).Seconds()
		start = time.Now()

		ker_in12 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-conv.csv", real_batch[0]*real_batch[1]*ker_size)
		ker_in12_0 := make([]float64, len(ker_in12)/2)
		ker_in12_1 := make([]float64, len(ker_in12)/2)
		if wide_case == 3 {
			for k := 0; k < ker_size; k++ {
				for i := 0; i < real_batch[0]; i++ {
					for j := 0; j < real_batch[1]/2; j++ {
						ker_in12_0[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+2*j)]   // [i][2*j]
						ker_in12_1[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+2*j+1)] // [i][2*j+1]
					}
				}
			}
		}

		bn_a := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-a.csv", real_batch[1])
		bn_b := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-b.csv", real_batch[1])

		if wide_case == 2 {
			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in12, bn_a, bn_b, alpha, pow, in_wids[0], raw_in_wids[1], ker_wid, real_batch[0], real_batch[1], norm[1], 0, step[1], 2, log_sparse[0]-1, "StrConv_sparse", fast_pack, debug, ark)
		} else if wide_case == 3 {
			bn_a_0 := make([]float64, real_batch[1]/2)
			bn_a_1 := make([]float64, real_batch[1]/2)
			bn_b_0 := make([]float64, real_batch[1]/2)
			bn_b_1 := make([]float64, real_batch[1]/2)
			for i := range bn_b_0 {
				bn_a_0[i] = bn_a[2*i]
				bn_a_1[i] = bn_a[2*i+1]
				bn_b_0[i] = bn_b[2*i]
				bn_b_1[i] = bn_b[2*i+1]
			}
			ct_result1 := evalConv_BNRelu_new(cont, ct_layer, ker_in12_0, bn_a_0, bn_b_0, alpha, pow, in_wids[0], raw_in_wids[1], ker_wid, real_batch[0], real_batch[1]/2, norm[0], 0, step[1], 2, 0, "StrConv_sparse_full", fast_pack, debug, ark)
			ct_result2 := evalConv_BNRelu_new(cont, ct_layer, ker_in12_1, bn_a_1, bn_b_1, alpha, pow, in_wids[0], raw_in_wids[1], ker_wid, real_batch[0], real_batch[1]/2, norm[0], 0, step[1], 2, 0, "StrConv_sparse_full", fast_pack, debug, ark)

			xi := make([]float64, cont.N)
			xi[2] = 1.0
			xi_plain := ckks.NewPlaintext(cont.params, ct_result2.Level(), 1.0)
			cont.encoder.EncodeCoeffs(xi, xi_plain)
			cont.encoder.ToNTT(xi_plain)
			ct_result2 = cont.evaluator.MulNew(ct_result2, xi_plain)
			ct_layer = cont.evaluator.AddNew(ct_result1, ct_result2)
		}
		fmt.Println("Block1 to 2 done!")
		timings[1] = time.Since(start).Seconds()
		start = time.Now()

		// ResNet Block 2
		for i := 1; i <= num_blcs[1]; i++ {
			if i == 5 {
				pow = init_pow
			}
			bn_a2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-a.csv", real_batch[1])
			bn_b2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-b.csv", real_batch[1])
			ker_in2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-conv.csv", real_batch[1]*real_batch[1]*ker_size)

			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in2, bn_a2, bn_b2, alpha, pow, in_wids[1], raw_in_wids[1], ker_wid, real_batch[1], real_batch[1], norm[1], 0, step[1], 2, log_sparse[1], "Conv_sparse", fast_pack, debug, ark)
			fmt.Println("Block2, Layer ", i, "done!")
		}
		fmt.Println("Block2 done.")
		timings[2] = time.Since(start).Seconds()
		start = time.Now()

		pow = mid_pow
		ker_in23 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-conv.csv", real_batch[1]*real_batch[2]*ker_size)
		bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-a.csv", real_batch[2])
		bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-b.csv", real_batch[2])

		ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in23, bn_a3, bn_b3, alpha, pow, in_wids[1], raw_in_wids[2], ker_wid, real_batch[1], real_batch[2], norm[2], 0, step[2], 2, log_sparse[1]-1, "StrConv_sparse", fast_pack, debug, ark)
		fmt.Println("Block2 to 3 done!")
		timings[3] = time.Since(start).Seconds()
		start = time.Now()

		// ResNet Block 3
		for i := 1; i <= num_blcs[2]; i++ {
			if i == 3 {
				pow = init_pow
			}
			if i == 5 {
				pow = mid_pow
			}
			bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-a.csv", real_batch[2])
			bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-b.csv", real_batch[2])
			ker_in3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-conv.csv", real_batch[2]*real_batch[2]*ker_size)

			if i == num_blcs[2] {
				pow = final_pow
			}
			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in3, bn_a3, bn_b3, alpha, pow, in_wids[2], raw_in_wids[2], ker_wid, real_batch[2], real_batch[2], norm[2], 0, step[2], 2, log_sparse[2], "Conv_sparse", fast_pack, debug, ark)
			fmt.Println("Block3, Layer ", i, "done!")
		}
		fmt.Println("Block3 done.")
		timings[4] = time.Since(start).Seconds()
		start = time.Now()

		ker_inf_wid := raw_in_wids[2]
		if ker_inf_wid%2 == 0 {
			ker_inf_wid++
		}
		ker_inf := readTxt(weight_dir+"final-fckernel.csv", real_batch[2]*fc_out)

		ker_inf_ := make([]float64, ker_inf_wid*ker_inf_wid*real_batch[2]*fc_out)
		for i := range ker_inf {
			for b := 0; b < ker_inf_wid*ker_inf_wid; b++ {
				ker_inf_[i+b*real_batch[2]*fc_out] = ker_inf[i]
			}
		}
		bn_af := make([]float64, fc_out)
		for i := range bn_af {
			bn_af[i] = 1.0 / float64(raw_in_wids[2]*raw_in_wids[2])
		}
		bn_bf := readTxt(weight_dir+"final-fcbias.csv", fc_out)

		ct_result := evalConv_BN(cont, ct_layer, ker_inf_, bn_af, bn_bf, in_wids[2], ker_inf_wid, real_batch[2], fc_out, norm[2], float64(1<<30), false, ark)
		fmt.Println("Final FC done.")
		timings[5] = time.Since(start).Seconds()

		start = time.Now()
		cont.decryptor.Decrypt(ct_result, pl_input)
		res_tmp := cont.encoder.DecodeCoeffs(pl_input)
		t = time.Since(start).Seconds()
		avgTime[6] += t
		res_out := prt_mat_one_norm(res_tmp, max_batch[2], norm[2], ker_inf_wid/2+1, ker_inf_wid/2+1)
		fmt.Println("\n Want: ", res_out[:fc_out])
		writeTxt(out_dir+"class_result_"+ker_name+"_"+strconv.Itoa(iter)+".csv", res_out[:fc_out])

		count := iter + 1
		setupavg := avgTime[0] / float64(1)
		encryptevg := avgTime[1] / float64(count)
		kernelavg := avgTime[2] / float64(count)
		convavg := avgTime[3] / float64(count)
		bootavg := avgTime[4] / float64(count)
		reluavg := avgTime[5] / float64(count)
		decryptavg := avgTime[6] / float64(count)

		ot := (convavg + bootavg + reluavg)
		minutes := int(ot) / 60
		seconds := int(ot) % 60

		tt := (setupavg + encryptevg + kernelavg + convavg + bootavg + reluavg + decryptavg)
		minutes1 := int(tt) / 60
		seconds1 := int(tt) % 60
		runtime.GC()

		fmt.Printf("\nSetup Avg : %.6fs,   Encrypt Avg: %.6fs,  Kernel Avg: %.6fs,  Conv Avg: %.6fs,   Boot Avg: %.6fs,   Relu Avg : %.6fs,  Decrypt Avg: %.6fs \n", setupavg, encryptevg, kernelavg, convavg, bootavg, reluavg, decryptavg)
		fmt.Printf("Execution time: %02d:%02d, Total time (include setup, enc & dec): %02d:%02d , Memory: %.2f GB\n", minutes, seconds, minutes1, seconds1, bToGb(allocMem))
	}
}
