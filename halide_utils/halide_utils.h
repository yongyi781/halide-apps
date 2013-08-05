#pragma once

#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
#include <Halide.h>

using Halide::Image;
#include <image_io.h>

namespace halide_utils
{
	// Global variable declarations
	Halide::Var x("x"), y("y"), c("c");

	// Returns the average number of seconds a function takes over iterations iterations.
	template<typename F0>
	double timing(F0 f, int iterations = 1)
	{
		auto start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < iterations; ++i)
			f();
		auto end = std::chrono::high_resolution_clock::now();
		double p = (double)std::chrono::high_resolution_clock::period::num / std::chrono::high_resolution_clock::period::den;
		return (end - start).count() * p / iterations;
	}

	// Prints above timing (with way more precision than needed or accurate).
	template<typename F0>
	void printTiming(F0 f, std::string message = "", int iterations = 1)
	{
		if (!message.empty())
			std::cout << message;
		double t = timing(f, iterations);
		std::cout << std::setprecision(15) << t << " s " << std::endl;
	}

	// Loads an image and outputs the time it took, with an option to gamma-correct.
	template<typename T>
	Halide::Image<T> load2(std::string fileName = "Images/in.png", bool gammaCorrect = false)
	{
		Halide::Image<T> im;
		printTiming([&] { im = load<T>(fileName); }, "Loading " + fileName + "... ");
		if (!gammaCorrect)
			return im;
		Halide::Func f;
		f(x, y) = pow(im(x, y), 2.2f);
		return (Halide::Image<T>)f.realize(im.width(), im.height(), im.channels());
	}

	// Saves an image and outputs the time it took, with an option to gamma-correct.
	template<typename T>
	void save2(const Halide::Image<T>& im, std::string fileName = "Images/out.png", bool gammaCorrect = false)
	{
		if (!gammaCorrect)
			printTiming([&] { save(im, fileName); }, "Saving to " + fileName + "... ");
		else
		{
			Halide::Func f;
			f(x, y) = pow(im(x, y), 1 / 2.2f);
			printTiming([&] { save((Halide::Image<T>)f.realize(im.width(), im.height(), im.channels()), fileName); }, "Saving to " + fileName + "... ");
		}
	}

	// Constructs a Func with infinite domain from an image, clipping to edges.
	template<typename T>
	Halide::Func clipToEdges(const Halide::Image<T>& im)
	{
		Halide::Func f;
		f(x, y) = im(clamp(x, 0, im.width() - 1), clamp(y, 0, im.height() - 1));
		return f;
	}

	// Constructs a Func with infinite domain from an ImageParam, clipping to edges.
	Halide::Func clipToEdges(const Halide::ImageParam& im)
	{
		Halide::Func f;
		f(x, y) = im(clamp(x, 0, im.width() - 1), clamp(y, 0, im.height() - 1));
		return f;
	}

	// Constructs a Func with infinite domain from an image, clipping to black.
	template<typename T>
	Halide::Func clipToBlack(const Halide::Image<T>& im)
	{
		Halide::Func f;
		f(x, y) = select(x >= 0 && x < im.width() && y >= 0 && y < im.height(),
			im(clamp(x, 0, im.width() - 1), clamp(y, 0, im.height() - 1)), T());
		return f;
	}

	template<typename T1, typename T2, typename T3, size_t M, size_t N>
	void matrixMultiply(const T1(&m)[M][N], const T2(&v)[N], T3(&out)[M])
	{
		for (size_t i = 0; i < M; ++i)
		{
			out[i] = T3();
			for (size_t j = 0; j < N; ++j)
				out[i] += m[i][j] * v[j];
		}
	}

	template<typename T1, typename T2, size_t M, size_t N>
	void matrixMultiply(const T1(&m)[M][N], const T2(&v)[N], Halide::Expr(&out)[M])
	{
		for (size_t i = 0; i < M; ++i)
		{
			out[i] = 0.f;
			for (size_t j = 0; j < N; ++j)
				out[i] += m[i][j] * v[j];
		}
	}

	template<typename T>
	Halide::Func brightness(Halide::Func in, T factor)
	{
		Halide::Func f;
		f(x, y, c) = in(x, y, c) * factor;
		return f;
	}

	Halide::Func contrast(Halide::Func in, float factor, float midpoint = 0.3f)
	{
		return (Halide::Func)(factor * in + (1.f - factor) * midpoint);
	}

	Halide::Func BW(Halide::Func in, float wr = 0.3f, float wg = 0.6f, float wb = 0.1f, bool output2D = false)
	{
		Halide::Func f;
		Halide::Expr e = wr * in(x, y, 0) + wg * in(x, y, 1) + wb * in(x, y, 2);
		if (output2D)
			f(x, y) = e;
		else
			f(x, y, c) = e;
		return f;
	}

	std::pair<Halide::Func, Halide::Func> lumiChromi(Halide::Func in, float wr = 0.3f, float wg = 0.6f, float wb = 0.1f)
	{
		Halide::Func lumi = BW(in, wr, wg, wb);
		return std::make_pair(lumi, (Halide::Func)(in / lumi));
	}

	Halide::Func brightnessContrastLumi(Halide::Func in, float brightF, float contrastF, float midpoint = 0.3f)
	{
		auto yc = lumiChromi(in);
		Halide::Func y = brightness(contrast(yc.first, contrastF, midpoint), brightF);
		return (Halide::Func)(y * yc.second);
	}

	const float RGB_TO_YUV_MATRIX[3][3] = { { 0.299f, 0.587f, 0.114f }, { -0.14713f, -0.28886f, 0.436f }, { 0.615f, -0.51499f, -0.10001f } };
	const float YUV_TO_RGB_MATRIX[3][3] = { { 1.f, 0.f, 1.13983f }, { 1.f, -0.39465f, -0.58060f }, { 1.f, 2.03211f, 0.f } };

	Halide::Func rgbToYuv(Halide::Func in)
	{
		Halide::Expr rgb[3] = { in(x, y, 0), in(x, y, 1), in(x, y, 2) };
		Halide::Expr yuv[3];
		matrixMultiply(RGB_TO_YUV_MATRIX, rgb, yuv);
		Halide::Func f;
		f(x, y, c) = select(c == 0, yuv[0], select(c == 1, yuv[1], yuv[2]));
		return f;
	}

	Halide::Func yuvToRgb(Halide::Func in)
	{
		Halide::Expr yuv[3] = { in(x, y, 0), in(x, y, 1), in(x, y, 2) };
		Halide::Expr rgb[3];
		matrixMultiply(YUV_TO_RGB_MATRIX, yuv, rgb);
		Halide::Func f;
		f(x, y, c) = select(c == 0, rgb[0], select(c == 1, rgb[1], rgb[2]));
		return f;
	}

	Halide::Func saturate(Halide::Func in, float k)
	{
		Halide::Func yuv = rgbToYuv(in);
		Halide::Func f;
		f(x, y, c) = select(c == 0, yuv(x, y, 0), k * yuv(x, y, c));
		return yuvToRgb(f);
	}

	Halide::Expr interpolateNN(Halide::Func in, Halide::Expr x, Halide::Expr y)
	{
		return in(Halide::cast<int>(Halide::round(x)), Halide::cast<int>(Halide::round(y)));
	}

	Halide::Func scaleNN(Halide::Func in, float k)
	{
		Halide::Func f;
		f(x, y) = interpolateNN(in, x / k, y / k);
		return f;
	}

	Halide::Expr interpolateLin(Halide::Func in, Halide::Expr x, Halide::Expr y)
	{
		Halide::Expr xlow = Halide::cast<int>(x), ylow = Halide::cast<int>(y);
		Halide::Expr topInterp = (1 + xlow - x) * in(xlow, ylow) + (x - xlow) * in(xlow + 1, ylow);
		Halide::Expr bottomInterp = (1 + xlow - x) * in(xlow, ylow + 1) + (x - xlow) * in(xlow + 1, ylow + 1);
		return (1 + ylow - y) * topInterp + (y - ylow) * bottomInterp;
	}

	Halide::Func scaleLin(Halide::Func in, float k)
	{
		Halide::Func f;
		f(x, y) = interpolateLin(in, x / k, y / k);
		return f;
	}

	Halide::Func rotate(Halide::Func in, float centerx, float centery, float theta)
	{
		float c = cos(theta), s = sin(theta);
		Halide::Func f;
		f(x, y) = interpolateLin(in,
			centerx - (y - centery)*s + (x - centerx)*c,
			centery + (y - centery)*c + (x - centerx)*s);
		return f;
	}

	template<int Width, int Height>
	Halide::Image<float> toImage(const float (&arr)[Height][Width])
	{
		Halide::Image<float> im(Width, Height);
		for (int y = 0; y < Height; ++y)
			for (int x = 0; x < Width; ++x)
				im(x, y) = arr[y][x];
		return im;
	}

	Halide::Image<float> transpose(const Halide::Image<float>& im)
	{
		Halide::Image<float> transposed(im.height(), im.width());
		for (int yi = 0; yi < im.height(); ++yi)
			for (int xi = 0; xi < im.width(); ++xi)
				transposed(yi, xi) = im(xi, yi);
		return transposed;
	}

	void transposeInPlace(Halide::Image<float>& im)
	{
		for (int yi = 0; yi < im.height(); ++yi)
		{
			for (int xi = 0; xi < im.width(); ++xi)
			{
				float& f1 = im(xi, yi);
				float& f2 = im(yi, xi);
				std::swap(f1, f2);
			}
		}
	}

	namespace kernels
	{
		const float _deriv[][2] = { { -1, 1 } };
		const float _sobel[][3] = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
		const float _gauss3[][3] = { { 1.f / 16, 2.f / 16, 1.f / 16 }, { 2.f / 16, 4.f / 16, 2.f / 16 }, { 1.f / 16, 2.f / 16, 1.f / 16 } };
		const float _lap2d[][3] = { { 0, -1, 0 }, { -1, 4, -1 }, { 0, -1, 0 } };

		const Halide::Image<float> DERIV = toImage(_deriv);
		const Halide::Image<float> SOBEL = toImage(_sobel);
		const Halide::Image<float> SOBEL_Y = transpose(SOBEL);
		const Halide::Image<float> GAUSS3 = toImage(_gauss3);
		const Halide::Image<float> LAP2D = toImage(_lap2d);
	}

	Halide::Func convolve(Halide::Func in, Halide::Image<float> kernel)
	{
		Halide::Func f;
		Halide::RDom r(kernel);
		if (in.dimensions() >= 3)
			f(x, y, c) = Halide::sum(kernel(r.x, r.y) * in(x + r.x - kernel.width() / 2, y + r.y - kernel.height() / 2, c));
		else
			f(x, y) = Halide::sum(kernel(r.x, r.y) * in(x + r.x - kernel.width() / 2, y + r.y - kernel.height() / 2));
		return f;
	}

	Halide::Image<float> convolve(Halide::Image<float> im, Halide::Image<float> kernel)
	{
		Halide::Func clipped = clipToEdges(im);
		return convolve(clipped, kernel).realize(im.width(), im.height(), im.channels());
	}

	Halide::Image<float> boxBlurKernel(int size)
	{
		Halide::Image<float> out(size, size);
		float factor = 1.f / (size * size);
		for (int yi = 0; yi < size; ++yi)
			for (int xi = 0; xi < size; ++xi)
				out(xi, yi) = factor;
		return out;
	}

	Halide::Image<float> horiBlurKernel(int size)
	{
		Halide::Image<float> out(size, 1);
		float factor = 1.f / size;
		for (int xi = 0; xi < size; ++xi)
			out(xi) = factor;
		return out;
	}

	Halide::Func boxBlur(Halide::Func in, int size, bool use2DKernel = false)
	{
		if (use2DKernel)
			return convolve(in, boxBlurKernel(size));
		Halide::Image<float> kernel = horiBlurKernel(size);
		Halide::Func f = convolve(in, kernel);
		f.compute_root();
		return convolve(f, transpose(kernel));
	}

	Halide::Image<float> horiGaussKernel(float sigma)
	{
		Halide::Image<float> out(6 * (int)sigma + 1, 1);
		int center = 3 * (int)sigma;
		out(center) = 1.f;
		float total = 1.f;
		for (int xi = 1; xi <= center; ++xi)
			total += 2 * (out(center + xi) = out(center - xi) = exp(-xi * xi / (2.f * sigma * sigma)));
		for (int i = 0; i < out.width(); ++i)
			out(i) /= total;
		return out;
	}

	Halide::Image<float> gauss2D(float sigma)
	{
		Halide::Image<float> hori = horiGaussKernel(sigma);
		Halide::Image<float> out(hori.width(), hori.width());
		for (int yi = 0; yi < hori.width(); ++yi)
			for (int xi = 0; xi < hori.width(); ++xi)
				out(xi, yi) = hori(xi) * hori(yi);
		return out;
	}

	Halide::Func gaussianBlur(Halide::Func in, float sigma, bool use2DKernel = false)
	{
		if (use2DKernel)
			return convolve(in, gauss2D(sigma));
		Halide::Image<float> kernel = horiGaussKernel(sigma);
		Halide::Func f = convolve(in, kernel);
		f.compute_root();
		return convolve(f, transpose(kernel));
	}

	Halide::Func unsharpMask(Halide::Func in, float sigma, float strength, bool use2DKernel = false)
	{
		return (Halide::Func)(in + strength * (in - gaussianBlur(in, sigma, use2DKernel)));
	}

	Halide::Func laplacian(Halide::Func in)
	{
		return convolve(in, kernels::LAP2D);
	}

	Halide::Func basicGreen(Halide::Func in, int offset = 1)
	{
		Halide::Func f;
		f(x, y) = select((x + y + offset) % 2 == 0, in(x, y),
			(in(x + 1, y) + in(x - 1, y) + in(x, y + 1) + in(x, y - 1)) / 4);
		return f;
	}

	Halide::Func basicRorB(Halide::Func in, int offsetx, int offsety)
	{
		Halide::Func f;
		Halide::Expr nx = x - offsetx;
		Halide::Expr ny = y - offsety;
		f(x, y) = select(
			nx % 2 == 0 && ny % 2 == 0, in(x, y), select(
			nx % 1 == 1 && ny % 2 == 1,
			(in(x + 1, y + 1) + in(x - 1, y + 1) + in(x - 1, y - 1) + in(x + 1, y - 1)) / 4,
			select(ny % 2 == 0, (in(x + 1, y) + in(x - 1, y)) / 2, (in(x, y + 1) + in(x, y - 1)) / 2)));
		return f;
	}

	Halide::Func basicDemosaic(Halide::Func in, int offsetGreen = 1, int offsetRedX = 1, int offsetRedY = 1, int offsetBlueX = 0, int offsetBlueY = 0)
	{
		Halide::Func f;
		f(x, y, c) = select(c == 0, basicRorB(in, offsetRedX, offsetRedY)(x, y),
			select(c == 1, basicGreen(in, offsetGreen)(x, y), basicRorB(in, offsetBlueX, offsetBlueY)(x, y)));
		return f;
	}

	void prod(float M[3][3], Halide::Expr x, Halide::Expr y, Halide::Expr* x2, Halide::Expr* y2)
	{
		*x2 = M[0][0] * x + M[0][1] * y + M[0][2];
		*y2 = M[1][0] * x + M[1][1] * y + M[1][2];
		Halide::Expr w = M[2][0] * x + M[2][1] * y + M[2][2];
		*x2 /= w;
		*y2 /= w;
	}

	Halide::Func applyHomography(Halide::Image<float> source, Halide::Func dest, float H[3][3], bool bilinear = false)
	{
		Halide::Expr x2, y2;
		prod(H, x, y, &x2, &y2);
		Halide::Func clipped = clipToEdges(source);
		Halide::Expr sourceExpr = bilinear ? interpolateLin(clipped, x2, y2) : interpolateNN(clipped, x2, y2);
		Halide::Func f;
		f(x, y) = select(x2 >= 0 && x2 < source.width() && y2 >= 0 && y2 < source.height(), sourceExpr, dest(x, y));
		return f;
	}

	Halide::Func computeTensor(Halide::Func in, float sigmaG = 1.f, float factorSigma = 4.f)
	{
		Halide::Func bw = BW(in, 0.3f, 0.6f, 0.1f, true);
		Halide::Func l = gaussianBlur(bw, sigmaG);
		Halide::Func gradx = convolve(l, kernels::SOBEL);
		Halide::Func grady = convolve(l, kernels::SOBEL_Y);
		Halide::Func gradxx = lambda(x, y, gradx(x, y) * gradx(x, y));
		Halide::Func gradxy = lambda(x, y, gradx(x, y) * grady(x, y));
		Halide::Func gradyy = lambda(x, y, grady(x, y) * grady(x, y));
		Halide::Func f;
		f(x, y, c) = select(c == 0, gradxx(x, y), select(c == 1, gradxy(x, y), gradyy(x, y)));
		Halide::Func g = gaussianBlur(f, sigmaG * factorSigma);
		bw.compute_root();
		l.compute_root();
		f.compute_root();
		return g;
	}

	Halide::Func harrisCorners(Halide::Func in, int width, int height, float k = 0.15f, float sigmaG = 1.f, float factor = 4.f, int maxiDiam = 7, int boundarySize = 5)
	{
		Halide::Func tensor = computeTensor(in, sigmaG, factor);
		Halide::Func det("det");
		det(x, y) = tensor(x, y, 0) * tensor(x, y, 2) - tensor(x, y, 1) * tensor(x, y, 1);
		Halide::Func trace("trace");
		trace(x, y) = tensor(x, y, 0) + tensor(x, y, 2);
		Halide::Func cornerResponse = (Halide::Func)(det - k * trace * trace);
		Halide::RDom r(-maxiDiam / 2, maxiDiam, -maxiDiam / 2, maxiDiam);
		Halide::Func maximumWindow("maximumWindow");
		maximumWindow(x, y) = maximum(cornerResponse(x + r.x, y + r.y));
		Halide::Func showCorners("showCorners");
		showCorners(x, y, c) = select((cornerResponse(x, y) == maximumWindow(x, y) &&
			x > boundarySize && x < width - boundarySize && y > boundarySize && y < height - boundarySize), 1.f, 0.f);
		cornerResponse.compute_root();
		maximumWindow.compute_root();
		return showCorners;
	}

	// Poisson image editing

	Halide::Func naiveComposite(Halide::Func fg, Halide::Func bg, Halide::Func mask, int xSrc, int ySrc)
	{
		Halide::Func f;
		f(x, y) = select(mask(x - xSrc, y - ySrc) == 1.f, fg(x - xSrc, y - ySrc), bg(x, y));
		return f;
	}

	float dotIm(Halide::Image<float> im1, Halide::Image<float> im2)
	{
		float total = 0;
		for (int y = 0; y < im1.height(); ++y)
			for (int x = 0; x < im1.width(); ++x)
				for (int c = 0; c < im1.channels(); ++c)
					total += im1(x, y, c) * im2(x, y, c);
		return total;
	}

	template<class T>
	Halide::Expr dotIm(T im1, T im2, int width, int height, int channels = 3)
	{
		Halide::RDom r(0, width, 0, height, 0, channels);
		return Halide::sum(im1(r.x, r.y, r.z) * im2(r.x, r.y, r.z));
	}

	Halide::Image<float> ramp(int width, int height)
	{
		Halide::Image<float> im(width, height, 3);
		for (int y = 0; y < height; ++y)
			for (int x = 0; x < width; ++x)
				for (int c = 0; c < 3; ++c)
					im(x, y, c) = (float)x / width;
		return im;
	}

	Halide::Image<float> poissonNoHalide(Halide::Image<float> bg, Halide::Image<float> fg, Halide::Image<float> mask, int niter = 200)
	{
		int w = bg.width(), h = bg.height();
		Halide::Image<float> b = convolve(fg, kernels::LAP2D);
		Halide::Image<float> v(w, h, 3);
		for (int y = 0; y < h; ++y)
			for (int x = 0; x < w; ++x)
				for (int c = 0; c < 3; ++c)
					v(x, y, c) = (1.f - mask(x, y, c)) * bg(x, y, c);
		for (int i = 0; i < niter; ++i)
		{
			Halide::Image<float> Av = convolve(v, kernels::LAP2D);
			Halide::Image<float> r(w, h, 3);
			for (int y = 0; y < h; ++y)
				for (int x = 0; x < w; ++x)
					for (int c = 0; c < 3; ++c)
						r(x, y, c) = (b(x, y, c) - Av(x, y, c)) * mask(x, y, c);
			Halide::Image<float> Ar = convolve(r, kernels::LAP2D);
			float alpha = dotIm(r, r) / dotIm(r, Ar);
			std::cout << alpha << std::endl;
			Halide::Image<float> vnew(w, h, 3);
			for (int y = 0; y < h; ++y)
				for (int x = 0; x < w; ++x)
					for (int c = 0; c < 3; ++c)
						vnew(x, y, c) = v(x, y, c) + alpha * r(x, y, c);
			v = vnew;
		}
		return v;
	}

	Halide::Func poissonIteration(Halide::ImageParam ip, Halide::Func b, Halide::Func mask, int width, int height)
	{
		Halide::Func ipf = clipToEdges(ip);
		Halide::Func Av = convolve(ipf, kernels::LAP2D);
		Av.compute_root();
		Halide::Func r("r");
		r(x, y, c) = (b(x, y, c) - Av(x, y, c)) * mask(x, y, c);
		r.compute_root();
		Halide::Func Ar = convolve(r, kernels::LAP2D);
		Ar.compute_root();
		Halide::Func alpha = (Halide::Func)(dotIm(r, r, width, height) / dotIm(r, Ar, width, height));
		alpha.compute_root();
		Halide::Func vnew;
		vnew(x, y, c) = ip(x, y, c) + alpha * r(x, y, c);
		return vnew;
	}

	Halide::Image<float> poisson(Halide::Func bg, Halide::Func fg, Halide::Func mask, int width, int height, int niter = 200)
	{
		Halide::Func b = convolve(fg, kernels::LAP2D);
		b.compute_root();
		Halide::Func v;	// Solution
		Halide::ImageParam ip(Halide::Float(32), 3);
		Halide::Func poissonIter = poissonIteration(ip, b, mask, width, height);
		printTiming([&] { poissonIter.compile_jit(); }, "Compiling poissonIter... ");
		v(x, y, c) = (1.f - mask(x, y, c)) * bg(x, y, c);
		Halide::Image<float> im = v.realize(width, height, 3);
		for (int i = 0; i < niter; ++i)
		{
			ip.set(im);
			im = poissonIter.realize(width, height, 3);
		}
		return im;
	}

	void poissonCGIteration(Halide::ImageParam ip, Halide::ImageParam r0, Halide::ImageParam p, Halide::Func b, Halide::Func mask, int width, int height, Halide::Func* px1, Halide::Func* pr1, Halide::Func* pp1)
	{
		Halide::Func x0 = clipToEdges(ip);
		Halide::Func p0 = clipToEdges(p);
		Halide::Func Ap0 = convolve(p0, kernels::LAP2D);
		Halide::Func alpha = (Halide::Func)(dotIm(r0, r0, width, height) / dotIm(p0, Ap0, width, height));
		alpha.compute_root();
		(*px1)(x, y, c) = x0(x, y, c) + alpha() * p0(x, y, c);
		(*pr1)(x, y, c) = (r0(x, y, c) - alpha() * Ap0(x, y, c)) * mask(x, y, c);
		(*pr1).compute_root();
		Halide::Func beta = (Halide::Func)(dotIm(*pr1, *pr1, width, height) / dotIm(r0, r0, width, height));
		beta.compute_root();
		(*pp1)(x, y, c) = (*pr1)(x, y, c) + beta() * p0(x, y, c);
		(*pp1).compute_root();
	}

	Halide::Image<float> poissonCG(Halide::Func bg, Halide::Func fg, Halide::Func mask, int width, int height, int niter = 100)
	{
		Halide::ImageParam ip(Halide::Float(32), 3), rParam(Halide::Float(32), 3), pParam(Halide::Float(32), 3);
		Halide::Func b = convolve(fg, kernels::LAP2D);
		b.compute_root();
		Halide::Func x0;	// Solution
		x0(x, y, c) = (1.f - mask(x, y, c)) * bg(x, y, c);
		Halide::Func Ax0 = convolve(x0, kernels::LAP2D);
		Ax0.compute_root();
		Halide::Func r0("r0");
		r0(x, y, c) = (b(x, y, c) - Ax0(x, y, c)) * mask(x, y, c);
		Halide::Func p0("d0");
		p0(x, y, c) = r0(x, y, c);

		Halide::Image<float> im = x0.realize(width, height, 3);
		Halide::Image<float> ri = r0.realize(width, height, 3);
		Halide::Image<float> pi = p0.realize(width, height, 3);
		Halide::Func x1, r1, p1;
		poissonCGIteration(ip, rParam, pParam, b, mask, width, height, &x1, &r1, &p1);
		for (int i = 0; i < niter; ++i)
		{
			ip.set(im);
			rParam.set(ri);
			pParam.set(pi);
			im = x1.realize(width, height, 3);
			ri = r1.realize(width, height, 3);
			pi = p1.realize(width, height, 3);
		}
		return im;
	}

	Halide::Image<float> poissonComposite(Halide::Func bg, Halide::Func fg, Halide::Func mask, int srcWidth, int srcHeight, int srcX, int srcY, int width, int height, bool CG = true, bool useLog = true, int niter = 200)
	{
		if (useLog)
		{
			bg = (Halide::Func)log(bg + 1e-7f);
			fg = (Halide::Func)log(fg + 1e-7f);
		}
		Halide::Func bg2;
		bg2(x, y) = bg(x + srcX, y + srcY);
		Halide::Image<float> out;
		if (CG)
			out = poissonCG(bg2, fg, mask, width, height, niter);
		else
			out = poisson(bg2, fg, mask, width, height, niter);
		Halide::Func outClip = clipToBlack(out);
		Halide::Func f;
		f(x, y) = select(x >= srcX && y >= srcY && x < srcX + width && y < srcY + height, outClip(x - srcX, y - srcY), bg(x, y));
		if (useLog)
		{
			f = (Halide::Func)clamp(exp(f), 0.f, 1.f);
		}
		return f.realize(srcWidth, srcHeight, 3);
	}

	void rampTest(int niter = 0)
	{
		Halide::Image<float> maskIm = load2<float>("Images/Poisson/mask3.png");
		Halide::Image<float> rampIm = ramp(maskIm.width(), maskIm.height());
		Halide::Image<float> fgIm(maskIm.width(), maskIm.height(), 3);
		for (int y = 0; y < fgIm.height(); ++y)
			for (int x = 0; x < fgIm.width(); ++x)
				for (int c = 0; c < 3; ++c)
					fgIm(x, y, c) = 1.f;

		Halide::Func mask = clipToBlack(maskIm);
		Halide::Func fg = clipToBlack(fgIm);
		Halide::Func ramp = clipToBlack(rampIm);
		Halide::Image<float> out;
		printTiming([&] { out = poissonCG(ramp, fg, mask, maskIm.width(), maskIm.height(), niter); });

		save2(out);
	}

	namespace examples
	{
		void homographyExample(std::string bgImPath = "Images/pano/green.png", std::string posterImPath = "Images/pano/poster.png", std::string outFileName = "Images/out.png")
		{
			Halide::Image<float> im = load2<float>(bgImPath);
			Halide::Image<float> posterIm = load2<float>(posterImPath);
			Halide::Image<float> out;
			Halide::Func im1 = clipToEdges(im);
			float H[][3] = { { 0.8025f, 0.0116f, -78.2148f }, { -0.0058f, 0.8346f, -141.3292f }, { -0.0006f, -0.0002f, 1.f } };
			Halide::Func f = applyHomography(posterIm, im1, H, true);
			printTiming([&] { f.compile_jit(); }, "Compiling... ");
			printTiming([&] { out = f.realize(im.width(), im.height(), im.channels()); }, "Realizing... ");
			save2(out, outFileName);
		}

		void poissonExample(std::string bgImPath = "Images/pano/boston1-4.png", std::string bearImPath = "Images/Poisson/bear.png", std::string maskImPath = "Images/Poisson/mask.png", std::string outFileName = "Images/out.png")
		{
			Halide::Image<float> im = load2<float>(bgImPath);
			Halide::Image<float> bearIm = load2<float>(bearImPath);
			Halide::Image<float> maskIm = load2<float>(maskImPath);
			Halide::Image<float> out;

			Halide::Func pool = clipToEdges(im);
			Halide::Func bear = clipToEdges(bearIm);
			Halide::Func mask = clipToBlack(maskIm);
			out = poissonComposite(pool, bear, mask, im.width(), im.height(), 1, 300, bearIm.width(), bearIm.height(), true, true, 200);
			save2(out, outFileName);
		}
	}
}
