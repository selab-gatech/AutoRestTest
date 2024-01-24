package youtube.api.model;

import java.util.List;

public class Playlist {

    private String id;

    /**
     * The snippet object contains basic details about the playlist, such as its title, description, and category.
     */
    private SearchResultSnippet snippet;

    private String defaultLanguage;

    private List<Video> videos;

    public Playlist(String id, SearchResultSnippet snippet, String defaultLanguage, List<Video> videos) {
        this.id = id;
        this.snippet = snippet;
        this.defaultLanguage = defaultLanguage;
        this.videos = videos;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public SearchResultSnippet getSnippet() {
        return snippet;
    }

    public void setSnippet(SearchResultSnippet snippet) {
        this.snippet = snippet;
    }

    public String getDefaultLanguage() {
        return defaultLanguage;
    }

    public void setDefaultLanguage(String defaultLanguage) {
        this.defaultLanguage = defaultLanguage;
    }

    public List<Video> getVideos() {
        return videos;
    }

    public void setVideos(List<Video> videos) {
        this.videos = videos;
    }
}
