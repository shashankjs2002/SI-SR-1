"""Run with an authenticated Earth Engine project; see the companion Colab notebook."""
import argparse
from geodiff_gan.data.gee_pairs import IndiaPairConfig, IndiaPairDownloader, export_datasets, export_geotiff_datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="Your Earth Engine-enabled Google Cloud project ID")
    parser.add_argument("--root", required=True, help="Persistent output folder; rerun with the same folder to resume")
    parser.add_argument("--authenticate", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--format", choices=("geotiff", "npz"), default="geotiff")
    args = parser.parse_args()
    config = IndiaPairConfig()
    if not args.export_only:
        import ee
        if args.authenticate:
            ee.Authenticate()
        ee.Initialize(project=args.project)
        IndiaPairDownloader(args.root, config).collect()
    exporter = export_geotiff_datasets if args.format == "geotiff" else export_datasets
    for path in exporter(args.root, config):
        print(path)


if __name__ == "__main__":
    main()
