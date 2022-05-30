use std::error::Error;
use std::io;

use std::process;

// parejas de valores
use itertools::Itertools;

// graficar
use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::style::{PointMarker, PointStyle};
use plotlib::view::ContinuousView;

// DBSCAN
use ndarray::{array, ArrayView, Axis};

use petal_clustering::{Dbscan, Fit};
use petal_neighbors::distance::Euclidean;

// test normalidad
use statrs::distribution::{ChiSquared, Continuous};

pub fn dw(values: &Vec<f64>) {
    // Durbin-Watson valores críticos para 1 variable independiente
    // de acuerdo a https://www3.nd.edu/~wevans1/econ30331/Durbin_Watson_tables.pdf

    /*let values = [
        -3.15576, -3.2217, 0.089467, 4.0565, 0.334698, -3.69827, 0.612897, 4.169863, 5.792764,
        2.070962, -0.962, 2.660326, -8.74974,
    ];
    */
    let mut sumxmy2 = 0.0;
    let mut sumsq = 0.0;
    for i in 1..values.len() {
        sumxmy2 = sumxmy2 + f64::powf(values[i] - values[i - 1], 2.0);
    }
    for i in 0..values.len() {
        sumsq = sumsq + f64::powf(values[i], 2.0);
    }
    let d = sumxmy2 / sumsq;
    println!(
        "\n Test de Durbin et Watson (supuesto de independencia de los residuos)\n estadístico DW = d = {}",
        d
    );
    if d >= 1.5 && d <= 2.5 {
        println!(" >> No hay autocorrelación y si se cumple el supuesto, vía regla general.");
    } else {
        println!(" >> Si hay autocorrelación y no se cumple el supuesto, vía regla general.");
    }
    println!(" Si d esta fuera de los límites de https://www3.nd.edu/~wevans1/econ30331/Durbin_Watson_tables.pdf  ");
    println!(" entonces hay correlación entre los residuales y no son independientes. No se cumpliría el supuesto.\n");
}

pub fn mean(values: &Vec<f64>) -> f64 {
    if values.len() == 0 {
        return 0f64;
    }

    return values.iter().sum::<f64>() / (values.len() as f64);
}

pub fn variance(values: &Vec<f64>) -> f64 {
    if values.len() == 0 {
        return 0f64;
    }

    let mean = mean(values);
    return values
        .iter()
        .map(|x| f64::powf(x - mean, 2 as f64))
        .sum::<f64>()
        / values.len() as f64;
}

pub fn covariance(x_values: &Vec<f64>, y_values: &Vec<f64>) -> f64 {
    if x_values.len() != y_values.len() {
        panic!("  Los vectores son de diferente tamaño.");
    }

    let length: usize = x_values.len();

    if length == 0usize {
        return 0f64;
    }

    let mut covariance: f64 = 0f64;
    let mean_x = mean(x_values);
    let mean_y = mean(y_values);

    for i in 0..length {
        covariance += (x_values[i] - mean_x) * (y_values[i] - mean_y)
    }

    return covariance / length as f64;
}

fn f_dbscan(x: &Vec<f64>, y: &Vec<f64>) {
    // Create a couple of iterators
    let x_iter = x.iter();
    let y_iter = y.iter();

    // Interleave x_iter with y_iter and group into tuples of size 2 using itertools
    let mut v = Vec::new();
    for (a, b) in x_iter.interleave(y_iter).tuples() {
        v.push((*a, *b)); // If I don't de-reference with *, I get Vec<(&f64, &f64)>
    }

    //println!("v= {:?} ",v); *****************

    let mut points = array![
        [1.0, 2.0],
        [2.0, 2.0],
        [2.0, 2.3],
        [8.0, 7.0],
        [8.0, 8.0],
        [25.0, 80.0]
    ];
    //points.row_mut(0)[0] = 6.66;
    let zeros = ArrayView::from(&[0.; 2]).into_shape((1, 2)).unwrap();

    // apendizar pares necesarios
    let tam: i32;

    tam = x.len() as i32 - 5;

    for _ in 1..tam {
        points.append(Axis(0), zeros);
    }
    // copiar todo x e y a points

    for i in 0..x.len() {
        points.row_mut(i)[0] = x[i];
        points.row_mut(i)[1] = y[i];
    }
    println!(">>> Datos que se pasaran a DBSCAN ...\n{:?}", points);

    // vectores para los clusters encontrados
    let mut yy: Vec<f64> = Vec::new();
    let mut xx: Vec<f64> = Vec::new();

    // radio de vecindad = 3.0
    // El número mínimo de puntos necesarios para formar una región densa = 2
    let clustering = Dbscan::new(3.0, 2, Euclidean::default()).fit(&points);

    println!(
        "\n ======= Clusters encontrados {:?}  =======",
        clustering.0.len()
    ); // two clusters found
    for i in 0..clustering.0.len() {
        println!(
            "\n -------> Puntos en cluster {} {:?}",
            i + 1,
            clustering.0[&i]
        ); // the first three points in Cluster 0
           //println!("Puntos en cluster 1 {:?}", clustering.0[&1]); // [8.0, 7.0] and [8.0, 8.0] in Cluster 1
           // formar los vectores para el ajuste
        for m in clustering.0[&i].iter() {
            //println!("** {}",m);

            xx.push(x[*m]);
            yy.push(y[*m]);
        }
        // crear modelo lineal
        fit(&xx, &yy);

        xx.clear();
        yy.clear();
    }
    println!("\n Puntos fuera de clusters {:?}", clustering.1); // [25.0, 80.0] doesn't belong to any cluster
}

pub fn fit(x_values: &Vec<f64>, y_values: &Vec<f64>) {
    let b1 = covariance(x_values, y_values) / variance(x_values);
    let b0 = mean(y_values) - b1 * mean(x_values);

    let pearson: f64;
    let concord: f64;
    let r_ajust: f64;
    let mut predictions = Vec::new();
    let mut residuales: Vec<f64> = Vec::new();
    let mut sum_error = 0f64;
    let mut mse = 0f64;
    let length = x_values.len();

    println!("  interseccion : {}", b0);
    println!("  coeficiente  : {}", b1);
    println!("  ===>    FC= {} * edad + ({}) ", b1, b0);

    pearson = covariance(&y_values, &x_values)
        / (f64::powf(variance(x_values), 0.5) * f64::powf(variance(y_values), 0.5));
    println!("  Pearson r= {} ", pearson);

    concord = f64::powf(pearson, 2 as f64);
    println!("  Coeficiente de concordancia R^2: {}", concord);

    if x_values.iter().len() > 0 {
        r_ajust = concord - (2.0 * (1.0 - concord) / (x_values.iter().len() as f64 - 2.0 - 1.0));
        println!("  R^2 ajustada : {} ", r_ajust);
    }

    println!("  n : {} ", x_values.iter().len());

    // predicciones
    for i in 0..x_values.len() {
        predictions.push(b0 + b1 * x_values[i]);
    }
    // suma de los errores cuadrados
    // mostrar residuales
    let media_est = mean(&predictions);
    for i in 0..length {
        mse += f64::powf(y_values[i] - media_est, 2.0); // equivalente a TSS
    }

    println!(" MSE : {} ", mse / length as f64);
    println!(" RMSE: {} ", f64::powf(mse, 0.5));
    for i in 0..length {
        sum_error += f64::powf(predictions[i] - y_values[i], 2f64); // equivalente a RSS
        residuales.push(predictions[i] - y_values[i]);
    }
    let r_dos = (mse - sum_error) / mse;
    println!(" R^2 : {}", r_dos);
    let r_ajustada = 1.0 - ((1.0 - r_dos) * (length as f64 - 1.0)) / (length as f64 - 2.0 - 1.0);
    println!(" r ajustada: {} ", r_ajustada);

    // mostrar los residuales
    println!(" Residuales: {:?} ", residuales);

    // probar el supuestos de que los residuales tienen distribución normal
    println!("\n Supuestos de los residuales a prueba:");
    dago_pear_k2(&mut residuales);
    dw(&residuales);
    residuales.clear();
    println!(" -------------- fin de prueba de supuestos de residuales\n");

    let mean_error = sum_error / (length as f64); // MLE  o MSE en español
                                                  // de acuerdo con https://www.iartificial.net/error-cuadratico-medio-para-regresion/
    println!(
        "  MSE : {}   Cuanto mayor sea este valor, peor es el modelo.",
        mean_error
    );
    // de acuerdo con https://sitiobigdata.com/2018/08/27/machine-learning-metricas-regresion-mse/#
    println!("  RMSE: {}", mean_error.sqrt());
    // max y min a considerar
    let ls = y_values.iter().cloned().fold(0. / 0., f64::max); //maximo
    let li = y_values.iter().cloned().fold(0. / 0., f64::min); //minimo
    println!(" fc superior: {} , fc inferior: {}", ls, li);
    // AIC
    /*let loglik = (-1.0 * length as f64) / 2.0
        * ((2.0 * std::f64::consts::PI).ln() + 1.0 + (mean_error).ln());
    let p = 2.0; // edad + 1       numero de variables independientes + 1 de error de varianza
    let aic = -2.0 * loglik + 2.0 * p;
    */

    // de acuerdo a:
    // https://stats.stackexchange.com/questions/261273/how-can-i-apply-akaike-information-criterion-and-calculate-it-for-linear-regress
    // https://www.sciencedirect.com/topics/mathematics/akaike-information-criterion equation (4.87)
    // Data fitting and regression
    // Xin-She Yang, in Introduction to Algorithms for Data Mining and Machine Learning, 2019
    let aic2: f64 =
        2.0 * 2.0 + x_values.iter().len() as f64 * (sum_error / x_values.iter().len() as f64).ln();
    println!(
        "  AIC:  {} el menor de todos los AIC es el mejor ajuste.\n",
        aic2
    );
}

pub fn scat(x: &Vec<f64>, y: &Vec<f64>, nom_file: String) {
    // crear pares
    // Create a couple of iterators
    let x_iter = x.iter();
    let y_iter = y.iter();

    // Interleave x_iter with y_iter and group into tuples of size 2 using itertools
    let mut v = Vec::new();
    for (a, b) in x_iter.interleave(y_iter).tuples() {
        v.push((*a, *b)); // If I don't de-reference with *, I get Vec<(&f64, &f64)>
    }
    // println!(">>>>> Pares    {:?}",v);

    let lsx = x.iter().cloned().fold(0. / 0., f64::max) + 1.0; //maximo
    let lix = x.iter().cloned().fold(0. / 0., f64::min) - 1.0; //minimo
    let lsy = y.iter().cloned().fold(0. / 0., f64::max) + 5.0; //maximo
    let liy = y.iter().cloned().fold(0. / 0., f64::min) - 5.0; //minimo

    // graficar pares *** ++++++++
    // scatter plot from the data
    let s1: Plot = Plot::new(v).point_style(
        PointStyle::new()
            .size(1.0)
            .marker(PointMarker::Circle) // setting the marker to be a square
            .colour("#DD3355"),
    ); // and a custom colour

    // Agregar otras graficas
    //*let data2 = vec![(-1.4, 2.5), (7.2, -0.3)];
    //*let s2: Plot = Plot::new(data2).point_style(
    //*    PointStyle::new() // uses the default marker
    //*        .colour("#35C788"),
    //*); // and a different colour

    // The 'view' describes what set of data is drawn
    let v = ContinuousView::new()
        .add(s1)
        // .add(s2)
        .x_range(lix, lsx)
        .y_range(liy, lsy)
        .x_label("Edad")
        .y_label("FC");

    // A page with a single view is then saved to an SVG file
    Page::single(&v).save(nom_file).unwrap();
    // crear pares fin *******************************************
}

use std::cmp::Ordering;
fn cmp_f64(a: &f64, b: &f64) -> Ordering {
    if a.is_nan() {
        return Ordering::Greater;
    }
    if b.is_nan() {
        return Ordering::Less;
    }
    if a < b {
        return Ordering::Less;
    } else if a > b {
        return Ordering::Greater;
    }
    return Ordering::Equal;
}
pub fn dago_pear_k2(x: &mut Vec<f64>) {
    x.sort_by(cmp_f64);

    //print!("ordenado {:?}", x);

    let alpha: f64 = 0.05;
    let n = x.len();
    let s1: f64 = x.iter().sum(); // suma de los elementos de x
                                  //println!("\ns1 ::: {}\n", s1);
    let s2: f64 = x.iter().map(|a| a.powi(2)).sum::<f64>(); //suma de cada elemento al cuadrado
    let s3: f64 = x.iter().map(|a| a.powi(3)).sum::<f64>(); //suma de cada elemento al cubo
    let s4: f64 = x.iter().map(|a| a.powi(4)).sum::<f64>(); //suma de cada elemento a la 4a

    //println!("\n n: {} s1 {} s2 {} s3 {} s4 {}", n, s1, s2, s3, s4);

    let ss: f64 = s2 - (f64::powf(s1, 2.0) / n as f64);
    let v: f64 = ss / (n as f64 - 1.0);
    let k3: f64 = ((n as f64 * s3) - (3.0 * s1 * s2) + ((2.0 * f64::powf(s1, 3.0)) / n as f64))
        / ((n as f64 - 1.0) * (n as f64 - 2.0));
    let g1: f64 = k3 / f64::powf(f64::powf(v, 3.0), 0.5);
    //println!("\nss {}  v {}  k3 {}   g1{}", ss, v, k3, g1);

    let k4: f64 = ((n as f64 + 1.0)
        * ((n as f64 * s4) - (4.0 * s1 * s3) + (6.0 * (f64::powf(s1, 2.0)) * (s2 / n as f64))
            - ((3.0 * (f64::powf(s1, 4.0))) / (f64::powf(n as f64, 2.0))))
        / ((n as f64 - 1.0) * (n as f64 - 2.0) * (n as f64 - 3.0)))
        - ((3.0 * (f64::powf(ss, 2.0))) / ((n as f64 - 2.0) * (n as f64 - 3.0)));

    let g2: f64 = k4 / f64::powf(v, 2.0);
    let eg1: f64 = ((n as f64 - 2.0) * g1) / f64::powf(n as f64 * (n as f64 - 1.0), 0.5);
    //let eg2: f64 = ((n as f64 - 2.0) * (n as f64 - 3.0) * g2)
    //    / ((n as f64 + 1.0) * (n as f64 - 1.0))
    //    + ((3.0 * (n as f64 - 1.0)) / (n as f64 + 1.0));
    //println!("\nk4 {}  g2 {}   eg1 {} ", k4, g2, eg1);

    let a: f64 = eg1
        * f64::powf(
            ((n as f64 + 1.0) * (n as f64 + 3.0)) / (6.0 * (n as f64 - 2.0)),
            0.5,
        );
    let b: f64 = (3.0
        * ((f64::powf(n as f64, 2.0)) + (27.0 * n as f64) - 70.0)
        * ((n as f64 + 1.0) * (n as f64 + 3.0)))
        / ((n as f64 - 2.0) * (n as f64 + 5.0) * (n as f64 + 7.0) * (n as f64 + 9.0));
    let c: f64 = f64::powf(2.0 * (b - 1.0), 0.5) - 1.0;
    let d: f64 = f64::powf(c, 0.5);
    let e: f64 = 1.0 / f64::powf(d.ln(), 0.5);
    let f: f64 = a / f64::powf(2.0 / (c - 1.0), 0.5);
    //println!("a {}  b {}  c {}  d {} e {} f{}", a, b, c, d, e, f);
    let zg1: f64 = e * (f + f64::powf(f64::powf(f, 2.0) + 1.0, 0.5)).ln();
    let g: f64 = (24.0 * n as f64 * (n as f64 - 2.0) * (n as f64 - 3.0))
        / (f64::powf(n as f64 + 1.0, 2.0) * (n as f64 + 3.0) * (n as f64 + 5.0));
    let h: f64 = ((n as f64 - 2.0) * (n as f64 - 3.0) * g2.abs())
        / ((n as f64 + 1.0) * (n as f64 - 1.0) * f64::powf(g, 0.5));
    let j: f64 = ((6.0 * (f64::powf(n as f64, 2.0) - (5.0 * n as f64) + 2.0))
        / ((n as f64 + 7.0) * (n as f64 + 9.0)))
        * f64::powf(
            (6.0 * (n as f64 + 3.0) * (n as f64 + 5.0))
                / (n as f64 * (n as f64 - 2.0) * (n as f64 - 3.0)),
            0.5,
        );
    //println!("\nzg1 {} g {}  h {}  j {}", zg1, g, h, j);
    let k: f64 = 6.0 + ((8.0 / j) * ((2.0 / j) + f64::powf(1.0 + (4.0 / f64::powf(j, 2.0)), 0.5)));
    let l: f64 = (1.0 - (2.0 / k)) / (1.0 + h * f64::powf(2.0 / (k - 4.0), 0.5));
    let zg2: f64 =
        (1.0 - (2.0 / (9.0 * k)) - f64::powf(l, 1. / 3.0)) / f64::powf(2.0 / (9.0 * k), 0.5);
    let k2: f64 = f64::powf(zg1, 2.0) + f64::powf(zg2, 2.0); // D'Agostino-Pearson statistic
                                                             //print!("\nk {} l {} zg2 {} k2: {}", k, l, zg2, k2);
                                                             //println!("\n k2 {}", k2);
    let x2: f64 = k2;
    let df: f64 = 2.0;

    let nn = ChiSquared::new(df).unwrap();
    let prob: f64 = nn.pdf(x2) * 2.0;
    println!("\n D'Agostino-Pearson normality test\n K2 is distributed as Chi-squared with df=2");
    println!(" k2 {}        p {}", x2, prob);

    if prob >= alpha {
        println!(" Tiene distribucion normal\n");
    } else {
        println!(" NO tiene distribucion normal\n");
    }
}

fn analisis() -> Result<(), Box<dyn Error>> {
    // Crear el lector CSV e iterar sobre cada registro.
    let mut rdr = csv::Reader::from_reader(io::stdin());
    let mut y: Vec<f64> = Vec::new();
    let mut x: Vec<f64> = Vec::new();

    for result in rdr.records() {
        // The iterator yields Result<StringRecord, Error>, so we check the
        // error here.
        let record = result?;

        // convertir a flotante
        y.push(record[0].parse::<f64>().unwrap());
        x.push(record[1].parse::<f64>().unwrap());
        //print!("{:?}",y)   // mostrar el vector construido
    }
    //print!("{:?}",y);

    println!("  media de FC es {}", mean(&y));
    println!("  media de Edad es {}", mean(&x));
    println!("  varianza de FC es {}", variance(&y));
    println!("  varianza de Edad es {}", variance(&x));
    println!("  covarianza de FC y Edad es {}", covariance(&y, &x));

    // graficar todo
    scat(&x, &y, (&"gra_todo1.svg").to_string());
    // crear modelo
    fit(&x, &y);
    // separar información en grupos iniciales
    f_dbscan(&x, &y);

    Ok(())
}

fn main() {
    println!("    ");
    println!("  Modelo de Frecuencia Cardiaca, BUAP México, ");
    println!("  Versión 3, Abril de 2022");
    println!("  Autor:  Dr. Enrique R.P. Buendia Lozada  ");
    println!("\n\n Ejemplo de como usar: \n- usar MSDOS \n- los archivos: frec.exe, fc.csv (con los datos de edad y FC) ; deben estar en la misma carpeta \n- Escribir esto y oprimir intro: frec.exe < fc.csv > resultado.txt");
    println!("    ");
    if let Err(err) = analisis() {
        println!(
            "error en la actividad de ejecución de implementación: {}",
            err
        );
        process::exit(1);
    }
}
