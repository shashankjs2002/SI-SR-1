"""Run with an authenticated Earth Engine project; see the companion Colab notebook."""
import argparse
from geodiff_gan.data.gee_pairs import IndiaPairConfig, IndiaPairDownloader, export_geotiff_datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="Your Earth Engine-enabled Google Cloud project ID")
    parser.add_argument("--root", required=True, help="Persistent output folder; rerun with the same folder to resume")
    parser.add_argument("--authenticate", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    config = IndiaPairConfig()
    if not args.export_only:
        import ee
        if args.authenticate:
            ee.Authenticate()
        ee.Initialize(project=args.project)
        IndiaPairDownloader(args.root, config).collect()
    for path in export_geotiff_datasets(args.root, config):
        print(path)


if __name__ == "__main__":
    main()
