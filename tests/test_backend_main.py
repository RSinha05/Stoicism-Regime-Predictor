import unittest
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException

import backend.main as main


class _DummyScaler:
    def transform(self, features):
        self.last_features = features
        return features


class _DummyPCA:
    def transform(self, scaled):
        self.last_scaled = scaled
        return np.array([[1.23456, -0.98765]])


class _DummyModel:
    def predict(self, x_wundt):
        self.last_predict_input = x_wundt
        return np.array([1])

    def predict_proba(self, x_wundt):
        self.last_predict_proba_input = x_wundt
        return np.array([[0.1, 0.7, 0.1, 0.1]])


class BackendMainTests(unittest.TestCase):
    def test_root_returns_api_metadata(self):
        response = main.root()
        self.assertEqual(response["version"], "1.0.0")
        self.assertIn("NIFTY50 Stoic-HMM API is live", response["message"])

    def test_health_reports_label_map_size(self):
        with patch.object(main, "label_map", {0: "A", 1: "B", 2: "C"}):
            response = main.health()
            self.assertEqual(response, {"status": "ok", "model_states": 3})

    def test_get_history_returns_wrapped_history(self):
        sample_history = [{"year": "FY2020", "regime": "DESIRE"}]
        with patch.object(main, "history", sample_history):
            response = main.get_history()
            self.assertEqual(response, {"data": sample_history})

    def test_get_year_returns_color_and_description_for_existing_year(self):
        sample_history = [{"year": "FY2022", "regime": "DESIRE"}]
        with patch.object(main, "history", sample_history):
            response = main.get_year("FY2022")
            self.assertEqual(response["year"], "FY2022")
            self.assertEqual(response["regime"], "DESIRE")
            self.assertEqual(response["color"], main.REGIME_COLORS["DESIRE"])
            self.assertEqual(
                response["description"], main.REGIME_DESCRIPTIONS["DESIRE"]
            )

    def test_get_year_raises_404_for_missing_year(self):
        with patch.object(main, "history", [{"year": "FY2023", "regime": "FEAR"}]):
            with self.assertRaises(HTTPException) as context:
                main.get_year("FY1999")
            self.assertEqual(context.exception.status_code, 404)
            self.assertEqual(context.exception.detail, "Year FY1999 not found")

    def test_get_proxy_matrix_returns_wrapped_matrix(self):
        sample_proxy = [{"year": "FY2020", "desire": 0.1}]
        with patch.object(main, "proxy_matrix", sample_proxy):
            response = main.get_proxy_matrix()
            self.assertEqual(response, {"data": sample_proxy})

    def test_get_regimes_returns_all_expected_regimes(self):
        response = main.get_regimes()
        self.assertEqual(
            set(response.keys()), {"DESIRE", "FEAR", "PLEASURE", "DISTRESS"}
        )
        for regime in response:
            self.assertEqual(response[regime]["color"], main.REGIME_COLORS[regime])
            self.assertEqual(
                response[regime]["description"], main.REGIME_DESCRIPTIONS[regime]
            )

    def test_predict_builds_features_and_formats_response(self):
        dummy_scaler = _DummyScaler()
        dummy_pca = _DummyPCA()
        dummy_model = _DummyModel()
        dummy_label_map = {0: "DISTRESS", 1: "DESIRE", 2: "FEAR", 3: "PLEASURE"}

        payload = main.MarketInput(
            pe_median=31.0,
            eps_growth=0.12,
            de_median=6.5,
            pat_change=1.5,
            roe_median=16.0,
            ebitda_vol=8.0,
            mc_growth=0.15,
            roe_change=1.0,
        )

        with patch.multiple(
            main,
            scaler=dummy_scaler,
            pca=dummy_pca,
            model=dummy_model,
            label_map=dummy_label_map,
        ):
            result = main.predict(payload)

        self.assertEqual(result.regime, "DESIRE")
        self.assertEqual(result.valence, 1.235)
        self.assertEqual(result.arousal, -0.988)
        self.assertEqual(result.confidence, 0.7)
        self.assertEqual(result.color, main.REGIME_COLORS["DESIRE"])
        self.assertEqual(result.description, main.REGIME_DESCRIPTIONS["DESIRE"])
        self.assertEqual(
            result.probabilities,
            {"DISTRESS": 0.1, "DESIRE": 0.7, "FEAR": 0.1, "PLEASURE": 0.1},
        )

        expected_features = np.array([[0.12, -0.3, -0.8, -0.35]])
        self.assertTrue(np.allclose(dummy_scaler.last_features, expected_features))
        self.assertTrue(np.allclose(dummy_pca.last_scaled, expected_features))
        self.assertTrue(
            np.allclose(dummy_model.last_predict_input, np.array([[1.23456, -0.98765]]))
        )
        self.assertTrue(
            np.allclose(
                dummy_model.last_predict_proba_input, np.array([[1.23456, -0.98765]])
            )
        )


if __name__ == "__main__":
    unittest.main()
