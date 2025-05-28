const fs = require('fs');
const { parse } = require('csv-parse/sync'); // 同期的なCSVパーサー
const { RandomForestRegression } = require('ml-random-forest'); // RandomForestRegression をインポート

// --- 設定項目 ---
const CSV_FILE_PATH = 'top10_combined_df.csv';             // ★入力CSVファイルへのパス
const MODEL_SAVE_PATH = 'top10_rf_alm_model.json';        // ★保存するモデルファイル名
const TARGET_COLUMN = 'ALM';                        // ★目的変数のカラム名
const START_FEATURE_COLUMN = 'Weight';                 // ★説明変数の開始カラム名
const END_FEATURE_COLUMN = 'Pancreatic amylase';                   // ★説明変数の終了カラム名

/**
 * CSVファイルを読み込み、指定された特徴量と目的変数を抽出・前処理する関数
 * (この関数は前回の回答と同じですが、エラーハンドリング等を改善できます)
 */
function loadAndPreprocessData(filePath, startFeatureCol, endFeatureCol, targetCol) {
    console.log(`データ読み込み開始: ${filePath}`);
    let fileContent;
    try {
        fileContent = fs.readFileSync(filePath, 'utf-8');
    } catch (error) {
        console.error(`エラー: ファイル ${filePath} が見つからないか読み込めません。`);
        throw error;
    }

    const records = parse(fileContent, {
        columns: true,
        skip_empty_lines: true,
        trim: true,
        cast: true, // 数値に自動変換を試みる (ただし、より厳密な型チェック推奨)
        on_record: (record) => { // 各カラムが数値であることを保証 (NaNチェックのため)
            for (const key in record) {
                if (typeof record[key] === 'string' && record[key].trim() !== '' && !isNaN(Number(record[key]))) {
                    record[key] = Number(record[key]);
                } else if (typeof record[key] !== 'number') {
                     // 何もしないか、エラーにするか、デフォルト値にするか (ここでは文字列のまま残る可能性)
                }
            }
            return record;
        }
    });

    if (records.length === 0) {
        throw new Error("CSVファイルにデータがありません。");
    }

    const header = Object.keys(records[0]);
    const startIndex = header.indexOf(startFeatureCol);
    const endIndex = header.indexOf(endFeatureCol);

    if (startIndex === -1 || endIndex === -1) {
        throw new Error(`エラー: 特徴量カラム "${startFeatureCol}" または "${endFeatureCol}" がCSVヘッダーに見つかりません。`);
    }
    if (header.indexOf(targetCol) === -1) {
        throw new Error(`エラー: 目的変数カラム "${targetCol}" がCSVヘッダーに見つかりません。`);
    }
    if (startIndex > endIndex) {
        throw new Error(`エラー: 開始特徴量 "${startFeatureCol}" は終了特徴量 "${endFeatureCol}" より前に現れる必要があります。`);
    }

    const featureNames = header.slice(startIndex, endIndex + 1);
    console.log(`抽出された特徴量 (${featureNames.length}個): ${featureNames.join(', ')}`);
    console.log(`目的変数: ${targetCol}`);

    const X = [];
    const y = [];

    for (const record of records) {
        const featureValues = featureNames.map(name => record[name]); // record[name]が数値であることを期待
        const targetValue = record[targetCol]; // record[targetCol]が数値であることを期待

        // いずれかの値が数値でない場合 (NaN や undefined)、その行はスキップ
        if (featureValues.some(val => typeof val !== 'number' || isNaN(val)) || typeof targetValue !== 'number' || isNaN(targetValue)) {
            // console.warn("警告: 数値データでないかNaNが含まれるため、行をスキップしました:", record);
            continue;
        }
        X.push(featureValues);
        y.push(targetValue);
    }

    if (X.length === 0) {
        throw new Error("有効な学習データが見つかりませんでした。全ての指定カラムが数値データであることを確認してください。");
    }
    console.log(`データ読み込み完了。学習サンプル数: ${X.length}`);
    return { X, y, featureNames };
}

/**
 * メイン処理: ランダムフォレストモデルの学習と保存
 */
async function trainAndSaveRfModel() {
    try {
        // 1. データの読み込みと前処理
        const { X, y, featureNames } = loadAndPreprocessData(CSV_FILE_PATH, START_FEATURE_COLUMN, END_FEATURE_COLUMN, TARGET_COLUMN);

        // 2. ランダムフォレスト回帰モデルのオプション設定
        // これらのパラメータはデータセットに応じて調整が必要です
        const rfOptions = {
            seed: 42,                   // 乱数シード (結果の再現性のため)
            maxFeatures: 2,        // 各決定木で使用する特徴量の最大数 ('sqrt', 'log2', または数値)
                                        // 'sqrt' は全特徴量の平方根の数
            replacement: false,         // trueにすると復元抽出 (ブートストラップサンプリング)
            nEstimators: 150,           // 作成する決定木の数
            treeOptions: {              // 個々の決定木(CART)に渡すオプション
                maxDepth: 10,           // 各決定木の最大の深さ (過学習防止)
                minNumSamples: 5        // リーフノードを作成するのに必要な最小サンプル数
            },
            // useSampleBagging: true,  // デフォルトでtrue (ブートストラップサンプリングを使用)
        };
        console.log("\nRandom Forest Regressor モデルの学習を開始します...");
        console.log("使用するオプション:", JSON.stringify(rfOptions, null, 2));

        const regression = new RandomForestRegression(rfOptions);

        // 3. モデルの学習
        // RandomForestRegression の train メソッドは X (特徴量の配列の配列) と y (目的変数の配列) を取ります
        regression.train(X, y);
        console.log("モデルの学習が完了しました。");

        // 4. モデルの保存
        // ml-random-forest は export() メソッド (toJSONのエイリアス) を持ちます
        const modelJson = regression.toJSON();
        fs.writeFileSync(MODEL_SAVE_PATH, JSON.stringify(modelJson, null, 2));
        console.log(`\n学習済みモデルを ${MODEL_SAVE_PATH} に保存しました。`);
        console.log("プロセス完了。");

    } catch (error) {
        console.error("\nエラーが発生しました:", error.message);
        if (error.stack) {
            // console.error(error.stack); // 詳細なスタックトレースが必要な場合
        }
    }
}

// スクリプト実行
trainAndSaveRfModel();