package youtube.api.repository;

import youtube.api.model.*;

import java.text.ParseException;
import java.util.*;
import java.util.stream.Collectors;

import static youtube.data.PlaylistData.getPlaylistData;
import static youtube.api.util.RepositoryUtils.*;

public class YoutubePlaylistRepository {

    private static YoutubePlaylistRepository instance = null;
    private List<Playlist> playlists;

    private YoutubePlaylistRepository(List<Video> videos) throws ParseException {
        this.playlists = instantiate(videos);
    }

    public static YoutubePlaylistRepository getInstance(List<Video> videos) throws ParseException {
        if(instance == null) {
            instance = new YoutubePlaylistRepository(videos);
        }

        return instance;
    }

    private List<Playlist> instantiate(List<Video> videos) throws ParseException {
        return getPlaylistData(videos);
    }

    public List<SearchResult> processDefaultRequest(SearchRequest searchRequest, String partType, Date[] dates) {
        List<SearchResult> result = new ArrayList<>();
        if(searchRequest.getForDeveloper() == null || !searchRequest.getForDeveloper()) {
            result = playlists.stream()
                    .filter(x -> filterByChannelId(x, searchRequest.getChannelId()))
                    .filter(x -> filterByDates(x, dates))
                    .filter(x -> filterByQuery(x, searchRequest.getQ()))
                    .filter(x -> filterByRelevanceLanguage(x, searchRequest.getRelevanceLanguage()))
                    .filter(x -> filterByTopicId(x, searchRequest.getTopicId()))
                    .map(x -> convertToSearchResult(x, partType))
                    .collect(Collectors.toList());
        }
        return result;
    }

    public Playlist getPlaylistById(String playlistId) {
        return playlists.stream()
                .filter(x -> x.getId().equals(playlistId))
                .findFirst()
                .orElse(null);
    }

}
