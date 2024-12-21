import win32com.client

def modify_existing_part():
    try:
        # ソリッドワークスのアプリケーションに接続
        sw = win32com.client.Dispatch("SldWorks.Application")
        sw.Visible = True

        # 既存の部品ファイルを開く
        part_path = "C:\\SWAPI\\Part1.SLDPRT"
        part_doc = sw.OpenDoc6(part_path, 1, 0, "", 0, 0)  # 1 = Part document type
        if part_doc is None:
            print("Error: 部品ドキュメントが開けませんでした。")
            return

        # アクティブなドキュメントを取得
        part = sw.ActiveDoc
        if part is None:
            print("Error: アクティブなドキュメントが見つかりません。")
            return

        # スケッチ平面を選択
        part.Extension.SelectByID2("Front Plane", "PLANE", 0, 0, 0, False, 0, None, 0)

        # スケッチの作成
        part.SketchManager.InsertSketch(True)
        part.ClearSelection2(True)

        # 四角形のスケッチを作成（サイズが10mmの四角形）
        sketch_manager = part.SketchManager
        sketch_manager.CreateCenterRectangle(0, 0, 0, 0.005, 0.005)  # 5mmの半径で10mmのサイズ

        # スケッチを閉じる
        part.SketchManager.InsertSketch(False)

        # スケッチを押し出して立方体を作成（押し出しの距離は10mm）
        feature_manager = part.FeatureManager
        
        # 引数の型と順序を見直す
        result = feature_manager.FeatureExtrusion2(
            True,            # Push/pull direction (True = both directions)
            False,           # Direction type (False = Blind)
            False,           # Reverse direction (False = no reverse)
            0,               # Start depth (0 = default)
            0.01,            # End depth (10mm)
            0,               # Twist angle
            0.01,            # Thickness
            0,               # First end condition (0 = default)
            0,               # Second end condition (0 = default)
            0,               # Third end condition (0 = default)
            0,               # Fourth end condition (0 = default)
            False,           # Thin feature (False = no thin feature)
            False,           # Merge result (False = no merge)
            False,           # Cut (False = no cut)
            False,           # Use Default (False = no default)
            0,               # Sketch boundary (0 = default)
            0,               # Base sketch (0 = default)
            False,           # Show hidden edges (False = no)
            False,           # Reverse direction (False = no reverse)
            0,               # Sketch ID (0 = default)
            0                # External reference (0 = default)
        )

        if result is None:
            print("Error: 押し出しフィーチャーが作成されませんでした。")
            return

        # ファイルの保存
        part.SaveAs("C:\\SWAPI\\ModifiedPart.SLDPRT")
        
        print("部品ドキュメントが開かれ、変更が保存されました。")
        
    except Exception as e:
        print(f"Error: {e}")

modify_existing_part()
