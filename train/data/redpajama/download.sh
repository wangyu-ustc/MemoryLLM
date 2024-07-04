CC_SNAPSHOT="2023-14"
LANG="en"
PARTITION="head_middle"
BASE_URL="https://data.together.xyz/redpajama-data-v2/v1.0.0"

listings_tag="${LANG}-${CC_SNAPSHOT}-${PARTITION}"
mkdir listings
wget "${BASE_URL}/listings/${listings_tag}.txt" -O "listings/${listings_tag}.txt"
listings_file="listings/${listings_tag}.txt"

# download documents
while read line; do
  url="${BASE_URL}/documents/${line}.json.gz"
  dest="documents/${line}.json.gz"
  mkdir -p $(dirname $dest)
  wget "$url" -O "$dest"
done <"$listings_file"

