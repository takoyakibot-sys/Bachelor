using SolidWorks.Interop.sldworks;
using SolidWorks.Interop.swconst;
using System;

namespace MySolidWorksProject
{
    class Program
    {
        static void Main(string[] args)
        {
            // SolidWorksアプリケーションに接続
            SldWorks swApp = new SldWorks();
            swApp.Visible = true;

            // 新しい部品ドキュメントを開く
            ModelDoc2 swModel = (ModelDoc2)swApp.NewPart();
            if (swModel == null)
            {
                Console.WriteLine("Error: 新しい部品ドキュメントが開けませんでした。");
                return;
            }

            // スケッチを開始
            swModel.SketchManager.InsertSketch(true);

            // 外径の円を作成 (半径5mm)
            swModel.SketchManager.CreateCircle(0, 0, 0, 0.005, 0, 0);
            
            // 内径の円を作成 (半径4mm)
            swModel.SketchManager.CreateCircle(0, 0, 0, 0.004, 0, 0);

            // スケッチを終了
            swModel.SketchManager.InsertSketch(false);

            // 押し出しの深さを定義
            double length = 0.1; // 100mm

            // 押し出しの作成
            FeatureManager swFeatureManager = swModel.FeatureManager;
            Feature swExtrusion = swFeatureManager.FeatureExtrusion2(
                true,    // flip
                false,   // merge
                false,   // useAutoSelect
                (int)swEndConditions_e.swEndCondBlind, // startCondition
                (int)swEndConditions_e.swEndCondBlind, // endCondition
                length,    // depth (100mm)
                0, // 第二方向の深さはなし
                false,   // draftOutward
                false,   // draftBothDirections
                true,    // thinWall (内外径を使用)
                false,   // thinWallDirectionOutward
                0,    // wallThickness (使用しない)
                0, // secondWallThickness (使用しない)
                false,   // flipSideToCut
                false,   // reverseSketchOffset
                false,   // autoAdd
                false,   // reverseDirection
                false,   // capEnds
                false,   // useThickCaps
                false,   // removeEndCaps
                0, // capThickness
                0, // capThicknessDepth
                false    // thinWallEndsInward
            );

            if (swExtrusion == null)
            {
                Console.WriteLine("Error: 押し出しの作成に失敗しました。");
                return;
            }

            // ファイルを保存
            swModel.SaveAs(@"C:\SWAPI\pipe10x10x100mm.sldprt");

            // 完了メッセージ
            Console.WriteLine("10mm x 10mm x 100mmの鉄パイプが作成され、保存されました。");
        }
    }
}
