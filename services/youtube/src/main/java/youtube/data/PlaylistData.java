package youtube.data;

import youtube.api.model.Playlist;
import youtube.api.model.SearchResultSnippet;
import youtube.api.model.Video;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.TimeZone;

public class PlaylistData {

    public static List<Playlist> getPlaylistData(List<Video> videos) throws ParseException {
        List<Playlist> result = new ArrayList<>();

        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'");
        dateFormat.setTimeZone(TimeZone.getTimeZone("UTC"));

        //##### Playlist 1 #####

        List<Video> videosPlaylist1 = videos.subList(2,7);

        SearchResultSnippet snippetP1 = new SearchResultSnippet();
        snippetP1.setTitle("My favourite videos");
        snippetP1.setPublishedAt(dateFormat.parse("2020-04-01T20:18:39Z"));
        snippetP1.setLiveBroadcastContent(SearchResultSnippet.LiveBroadcastContent.NONE);
        snippetP1.setDescription("These are my favourite videos of YouTube.");
        snippetP1.setChannelTitle("My channel");
        snippetP1.setChannelId("UCxwvutsrqponmlkjihgfedc");
        snippetP1.setThumbnails(videosPlaylist1.get(0).getSnippet().getThumbnails());

        Playlist playlist1 = new Playlist("PLM6w4jagjNhE0rMyag22T9xEfifA3-IVW", snippetP1, "en", videosPlaylist1);

        //##### Playlist 2 #####

        List<Video> videosPlaylist2 = Arrays.asList(videos.get(1), videos.get(2), videos.get(7));

        SearchResultSnippet snippetP2 = new SearchResultSnippet();
        snippetP2.setTitle("Music!");
        snippetP2.setPublishedAt(dateFormat.parse("2020-04-10T17:12:10Z"));
        snippetP2.setLiveBroadcastContent(SearchResultSnippet.LiveBroadcastContent.NONE);
        snippetP2.setDescription("");
        snippetP2.setChannelTitle("My channel");
        snippetP2.setChannelId("UCxwvutsrqponmlkjihgfedc");
        snippetP2.setThumbnails(videosPlaylist2.get(0).getSnippet().getThumbnails());

        Playlist playlist2 = new Playlist("PLt6nqoGW2VDS5IzIvMFNbXBiPqAu0g8KE", snippetP2, "en", videosPlaylist2);

        result.add(playlist1);
        result.add(playlist2);

        return result;
    }
}
